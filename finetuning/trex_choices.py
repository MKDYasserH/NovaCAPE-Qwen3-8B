"""Fit the CLM action head on planner-labeled choices from separate T-Rex courses."""
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

TRAIN_SEEDS = range(100, 112)
DEV_SEEDS = range(300, 304)


def examples(seeds):
    from clm.schema import build_pairs
    from trex.backends import build_question
    from trex.engine import Game
    from trex.planner import Planner, snapshot

    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        game, planner, held = Game(seed), Planner(), "run"
        game.press_jump()
        for frame in range(3600):
            if game.crashed:
                game.restart()
                held = "run"
            if frame % 5 == 0:
                plan = planner.plan(snapshot(game, held), (1, 2), (1, 3), 1)
                state, questions = build_question(plan, "labeled")
                query, keys, texts = build_pairs(state, questions)["action"]
                rows.append((query, texts, keys.index(plan.best)))
                # Vary visited states while retaining the planner's target for every example.
                action = rng.choice(plan.safe_actions) if plan.safe_actions and rng.random() < 0.12 else plan.best
                held = "run" if action == "jump" else action
                if action == "jump" and not game.trex.jumping:
                    game.release_duck()
                    game.press_jump()
            if held == "duck" and not game.trex.speed_drop:
                game.press_duck()
            elif held == "run" and (game.trex.speed_drop or game.trex.ducking):
                game.release_duck()
            game.step()
    return rows


def fit(parent: Path, out: Path):
    from clm.embedder import Embedder
    from clm.heads import make_head

    train, dev = examples(TRAIN_SEEDS), examples(DEV_SEEDS)
    texts = list(dict.fromkeys(text for query, choices, _ in train + dev for text in (query, *choices)))
    ids = {text: i for i, text in enumerate(texts)}
    vectors, tokens = Embedder().embed(texts)
    vectors = torch.from_numpy(vectors).cuda()
    ck = torch.load(parent, weights_only=True, map_location="cpu")
    cfg = ck["cfg"]
    def head(key):
        model = make_head(cfg["width"], cfg["depth"], cfg["projection_dim"], cfg["activation"],
                          cfg["layernorm"], cfg["residual"], cfg["hidden_size"]).cuda()
        model.load_state_dict(ck[key])
        return model
    states, actions = head("state_head"), head("action_head")
    with torch.no_grad():
        queries = torch.cat([F.normalize(states(v), dim=-1) for v in vectors.split(512)])
    def pack(rows):
        return tuple(torch.tensor(t, device="cuda") for t in (
            [ids[q] for q, _, _ in rows], [[ids[a] for a in opts] for _, opts, _ in rows], [y for _, _, y in rows]))
    train_q, train_a, train_y = pack(train)
    dev_q, dev_a, dev_y = pack(dev)
    scale = torch.as_tensor(ck["logit_scale"]).exp().clamp(max=100).item()
    def logits(q, a):
        embedded = F.normalize(actions(vectors[a]), dim=-1)
        return scale * (queries[q, None, :] * embedded).sum(-1)

    torch.manual_seed(1234)
    optimizer = torch.optim.AdamW(actions.parameters(), lr=0.0001, weight_decay=0)
    counts = torch.bincount(train_y, minlength=3)
    weights = counts[train_y].float().reciprocal()
    history, best = [], None
    for step in range(161):
        if step in (0, 10, 20, 40, 80, 160):
            with torch.no_grad():
                scores = torch.cat([logits(q, a) for q, a in zip(dev_q.split(256), dev_a.split(256))])
                right = scores.argmax(1) == dev_y
                accuracy = right.float().mean().item()
                by_action = [right[dev_y == i].float().mean().item() for i in range(3)]
                loss = F.cross_entropy(scores, dev_y).item()
                history.append(dict(step=step, accuracy=accuracy, by_action=by_action, loss=loss))
                print(history[-1], flush=True)
                key = (min(by_action), accuracy, -loss)
                if best is None or key > best:
                    best = key
                    selected = step
                    torch.save(dict(state_head=ck["state_head"], action_head=actions.state_dict(),
                                    logit_scale=ck["logit_scale"], cfg=cfg, step=step), out / "best.pt")
        if step < 160:
            idx = torch.multinomial(weights, 96, replacement=True)
            loss = F.cross_entropy(logits(train_q[idx], train_a[idx]), train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actions.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
    report = dict(parent=str(parent), train_seeds=list(TRAIN_SEEDS), dev_seeds=list(DEV_SEEDS), train_rows=len(train),
                  dev_rows=len(dev), train_counts=counts.tolist(), dev_counts=torch.bincount(dev_y).tolist(),
                  unique_texts=len(texts), encoder_tokens=tokens, selected_step=selected, history=history,
                  prompt="labeled", target="exact planner action", selection="worst action accuracy, accuracy, then negative loss")
    (out / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, "/clm/examples/t_rex")
    fit(Path(sys.argv[1]), Path(sys.argv[2]))
