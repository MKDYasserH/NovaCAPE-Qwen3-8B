"""Contracts that keep the post-training comparisons interpretable. Run with unittest."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.outcome_ranking import CoordinateAdjustment, mixed_rollouts
from finetuning.general_post import domain_choices, mine_domain_choices, preference_loss
from finetuning.swe_outcomes import WINDOW, make_rollouts


class PostTrainingTests(unittest.TestCase):
    def test_domain_distractors_preserve_split_and_exclude_goal_and_duplicates(self):
        rows = [dict(source="household", split=split, group=i // 2)
                for split in ("train", "dev") for i in range(12)]
        pairs = np.array([[i, i] for i in range(len(rows))])
        for split in ("train", "dev"):
            choices = domain_choices(rows, pairs, split)["household"]
            for query, correct, *alternatives in choices:
                self.assertEqual(query, correct)
                self.assertEqual(len(set(alternatives)), 7)
                self.assertTrue(all(rows[j]["split"] == split and rows[j]["group"] != rows[query]["group"]
                                    for j in alternatives))

    def test_mining_handles_near_identical_templates(self):
        rows = [dict(source="shop", split="train", group=i // 2) for i in range(12)]
        pairs = np.array([[i, i] for i in range(len(rows))])
        torch.manual_seed(12)
        vectors = F.normalize(torch.ones(12, 4) + torch.randn(12, 4) * 0.002)
        identity = SimpleNamespace(embed_actions=lambda x: x, embed_states=lambda x: x)
        mined = mine_domain_choices(identity, vectors, rows, pairs)["shop"]
        self.assertEqual(mined.shape, (48, 9))
        for i, target, *alternatives in mined:
            self.assertEqual(i, target)
            self.assertEqual(len(set(alternatives)), 7)
            self.assertTrue(all(rows[j]["group"] != rows[i]["group"] for j in alternatives))

    def test_preference_ablation_ignores_review_margin(self):
        answer = torch.tensor([-1., 1.], requires_grad=True)
        review = torch.tensor([2., -2.], requires_grad=True)
        preference_loss(answer, review, False).backward()
        self.assertIsNone(review.grad)
        preference_loss(answer, review, True).backward()
        self.assertTrue(torch.isfinite(review.grad).all())
        self.assertTrue((review.grad < 0).all())

    def test_outcome_adjustment_survives_checkpoint_export(self):
        torch.manual_seed(1)
        heads = [torch.nn.Linear(11, 7) for _ in range(2)]
        inputs = torch.randn(2, 20, 11)
        projected = torch.stack([head(x) for head, x in zip(heads, inputs)])
        rms = projected.square().mean((1, 2), keepdim=True).sqrt()
        adjust = CoordinateAdjustment(7)
        with torch.no_grad():
            adjust.gain.normal_()
            adjust.bias.normal_()
        expected = adjust(projected, rms)
        ckpt = {name: {"out.weight": h.weight.detach().clone(), "out.bias": h.bias.detach().clone()}
                for name, h in zip(("state_head", "action_head"), heads)}
        adjust.fold_into(ckpt, rms)
        encoded = [F.normalize(F.linear(x, ckpt[k]["out.weight"], ckpt[k]["out.bias"]), dim=-1)
                   for x, k in zip(inputs, ("state_head", "action_head"))]
        torch.testing.assert_close((encoded[0] * encoded[1]).sum(-1), expected)

    def test_deepswe_pairs_use_only_allowed_tasks_and_actual_tail(self):
        rows = [dict(task_id=task, trajectory_id=f"{task}-{passed}", step_idx=step, reward=passed)
                for task in ("fit", "reserved") for passed in (0, 1) for step in (2, 0, 1)]
        self.assertEqual(mixed_rollouts(rows, {"fit"}, 2), [[([2, 0], False), ([5, 3], True)]])

    def test_external_split_and_short_rollout_padding(self):
        rows = [dict(task_id=str(i), trajectory_id=f"rollout-{i}", step_idx=0, reward=i % 2,
                     config="swebench_verified/train" if i < 20 else "swebench_verified/val")
                for i in range(25)]
        indices, weights, _, groups = make_rollouts(rows, set())
        self.assertEqual(indices.shape, (25, WINDOW))
        self.assertTrue((weights.sum(dim=1) == 1).all())
        self.assertTrue(set(groups["train"]).isdisjoint(groups["dev"]))
        self.assertEqual(set(groups["test"]), {str(i) for i in range(20, 25)})
        with self.assertRaisesRegex(ValueError, "prohibited"):
            make_rollouts(rows, {"7"})
        rows.append(dict(rows[0], trajectory_id="overlap", config="swebench_verified/val"))
        with self.assertRaisesRegex(ValueError, "task-disjoint"):
            make_rollouts(rows, set())


if __name__ == "__main__":
    unittest.main()
