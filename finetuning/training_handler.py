"""The training loop shared by the three stages. Each stage subclasses Trainer in its own *_training_handler.py."""
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch

from finetuning.batches import BatchStream
from finetuning.chunks import ChunkStore
from finetuning.environment import commit_volume, pick_device
from finetuning.evaluation import Evaluator
from finetuning.heads import ContrastiveHeads
from finetuning.loss import contrastive_loss, embed_batch
from schemas.training import (
    BatchStreamSettings,
    ChunkId,
    EvalReport,
    PairArrays,
    StageRun,
    TrainingConfig,
    TrainingSummary,
    TrainState,
)

BEST_CHECKPOINT = "best.pt"  # weights with the best validation score so far
LATEST_CHECKPOINT = "latest.pt"  # weights, optimizer and counters, to resume after a crash
LOG_FILE = "log.jsonl"


class Trainer(ABC):
    uses_hard_negatives = False  # True: train only on chunks that have hard negatives, and load them

    def __init__(self, run: StageRun, config: TrainingConfig):
        self.name = run.name
        self.settings = run.settings
        self.config = config
        self.device = pick_device()
        torch.manual_seed(config.seed)

        self.store = ChunkStore(run.data_dir)
        self.runs_dir = Path(run.data_dir) / config.runs_dir
        self.run_dir = self.runs_dir / run.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.training_chunks = self._select_training_chunks()

        self.model = ContrastiveHeads(config.head, config.loss).to(self.device)
        self.model.log_logit_scale.requires_grad_(self.settings.learn_temperature)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.settings.learning_rate, weight_decay=config.optimizer.weight_decay
        )
        self.evaluator = Evaluator(self.store, config.held_out, self.device)
        self.state = TrainState()
        self.started_at = time.time()
        self._restore()
        print(f"{run.name}: {len(self.training_chunks)} training chunks on {self.device}, {self.settings}", flush=True)

    # --- What each stage defines ---

    @abstractmethod
    def validation_score(self, report: EvalReport) -> float:
        """Lower is better. Picks best.pt and drives early stopping."""

    def epoch_batches(self) -> AbstractContextManager[Iterable[PairArrays]]:
        return self.qa_batches(self.settings.batch_size)

    def evaluate(self, full_eval: bool) -> EvalReport:
        return EvalReport(
            step=self.state.step,
            epoch=self.state.epoch,
            logit_scale=self.model.logit_scale().item(),
            minutes=self._minutes(),
            general=self.evaluator.general(self.model),
            hard_negatives=self.evaluator.hard_negatives(self.model) if full_eval else None,
        )

    # --- The loop ---

    def train(self) -> TrainingSummary:
        if self.state.step == 0:
            self._checkpoint(full_eval=True)  # score before any update, as a baseline
        while not self._finished():
            self._train_epoch()
            self.state.epoch += 1
            self._checkpoint(full_eval=True)
        return TrainingSummary(
            stage=self.name,
            step=self.state.step,
            epoch=self.state.epoch,
            best_score=self.state.best_score,
            minutes=self._minutes(),
        )

    def qa_batches(self, batch_size: int) -> BatchStream:
        # A new chunk order and new shuffles every epoch.
        epoch_seed = self.config.seed + self.state.epoch
        order = np.random.default_rng(epoch_seed).permutation(len(self.training_chunks))
        settings = BatchStreamSettings(
            chunks=[self.training_chunks[i] for i in order],
            chunks_per_pool=self.settings.chunks_per_pool,
            pool_repeats=self.settings.pool_repeats,
            batch_size=batch_size,
            with_hard_negatives=self.uses_hard_negatives,
            seed=epoch_seed,
        )
        return BatchStream(self.store, settings)

    def _train_epoch(self) -> None:
        with self.epoch_batches() as batches:
            for batch in batches:
                self._train_step(batch)
                if self.state.step % self.settings.eval_every_steps == 0:
                    self._checkpoint(full_eval=self._is_full_eval_step())
                if self._should_stop_early():
                    break

    def _train_step(self, batch: PairArrays) -> None:
        loss = contrastive_loss(embed_batch(self.model, batch.to_device(self.device)))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # only fills .grad; the weights change in optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.grad_clip_norm)
        self.optimizer.step()
        self.state.step += 1
        if self.state.step == 1 or self.state.step % self.config.log_every_steps == 0:
            print(
                f"step {self.state.step} loss {loss.item():.4f} scale {self.model.logit_scale().item():.1f} "
                f"{self._minutes():.1f} min",
                flush=True,
            )

    def _finished(self) -> bool:
        return self.state.epoch >= self.settings.max_epochs or self._should_stop_early()

    def _should_stop_early(self) -> bool:
        out_of_patience = self.state.evals_without_improvement >= self.settings.patience
        out_of_steps = self.settings.max_steps is not None and self.state.step >= self.settings.max_steps
        return out_of_patience or out_of_steps

    def _is_full_eval_step(self) -> bool:
        every = self.settings.full_eval_every_steps
        return every is None or self.state.step % every == 0

    # --- Checkpoints ---

    def _checkpoint(self, full_eval: bool) -> None:
        report = self.evaluate(full_eval)
        score = self.validation_score(report)
        if score < self.state.best_score - self.config.min_improvement:
            self.state.best_score = score
            self.state.evals_without_improvement = 0
            torch.save(self._weights(), self.run_dir / BEST_CHECKPOINT)
            report.is_best = True
        else:
            # The original never counted these, so its early stopping could never trigger.
            self.state.evals_without_improvement += 1
        self._log(report)
        resumable = {**self._weights(), "optimizer": self.optimizer.state_dict(), "train_state": self.state.model_dump()}
        torch.save(resumable, self.run_dir / LATEST_CHECKPOINT)
        commit_volume()

    def _weights(self) -> dict:
        return {
            **self.model.checkpoint_weights(),
            "step": self.state.step,
            "epoch": self.state.epoch,
            "stage": self.name,
            "head": self.config.head.model_dump(),
            "settings": self.settings.model_dump(),
        }

    def _restore(self) -> None:
        latest = self.run_dir / LATEST_CHECKPOINT
        if latest.exists():
            checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
            self.model.load_checkpoint_weights(checkpoint)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.state = TrainState.model_validate(checkpoint["train_state"])
            print(f"resumed {self.name} at step {self.state.step}, epoch {self.state.epoch}", flush=True)
        elif self.settings.init_from_run:
            start = self.runs_dir / self.settings.init_from_run / BEST_CHECKPOINT
            self.model.load_checkpoint_weights(torch.load(start, map_location="cpu", weights_only=True))
            print(f"{self.name} starts from {start}", flush=True)

    def _select_training_chunks(self) -> list[ChunkId]:
        held_out = self.config.held_out.all_chunks
        chunks = [chunk for chunk in self.store.list_chunks() if chunk not in held_out]
        if self.uses_hard_negatives:
            chunks = [chunk for chunk in chunks if self.store.has_hard_negatives(chunk)]
        if not chunks:
            raise FileNotFoundError(f"no training chunks under {self.store.qa_dir}")
        return chunks

    def _log(self, report: EvalReport) -> None:
        line = report.model_dump_json(exclude_none=True)
        print(line, flush=True)
        with (self.run_dir / LOG_FILE).open("a") as log_file:
            log_file.write(line + "\n")

    def _minutes(self) -> float:
        return (time.time() - self.started_at) / 60
