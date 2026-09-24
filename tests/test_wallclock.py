"""CPU-only real rollout and mocked policy-supervision checks; no model/data load."""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from hasmem.engines import wallclock as run
from hasmem.core import Memory, ResidualProjection, digest_tensor


class CPUOnlyTorch:
    """Redirect only the runner's explicitly CUDA tensor constructors to CPU."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.zeros(*args, **kwargs)

    def tensor(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.tensor(*args, **kwargs)


def fake_experiment():
    # Intentionally bypass BaseExperiment.__init__: never load an LLM or data.
    ex = object.__new__(run.Experiment)
    ex.seed = 13
    ex.memory = Memory(8)
    ex.memory.risk = nn.Sequential(nn.Linear(132, 64), nn.Tanh(), nn.Linear(64, 2))
    nn.init.zeros_(ex.memory.risk[-1].weight)
    nn.init.zeros_(ex.memory.risk[-1].bias)
    ex.actions = dict(keep=0, shrink=0, expand=0)
    ex.search_counts = dict(owners=0, positive=0, no_positive=0, evaluations=0,
                            qualified_candidates=0, keep_penalty_terms=0,
                            position_buckets=dict(early=0, middle=0, late=0),
                            keep_streak_buckets=dict(**{"0": 0, "3": 0, "8plus": 0}),
                            length_buckets=dict(short=0, medium=0, long=0))
    ex.maintenance = 0
    ex.large_model = False
    ex.arm = "control"
    ex.keep_rate_ema = 1.0
    ex.prompt_slots = 4
    ex.unkeep_ema = 0.0
    ex.policy_started = None
    ex.policy_span_seconds = 1.0
    return ex


def stream(length, width=10):
    return [
        {"values": (torch.arange(width * 8, dtype=torch.float32).reshape(width, 8) + i),
         "prefix": [11], "suffix": [12], "body_ids": list(range(width))}
        for i in range(length)
    ]


class RunnerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.cpu = patch.object(run, "torch", CPUOnlyTorch())
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.ex = fake_experiment()

    def test_keep_streak_is_causal_and_capped(self):
        encoded = stream(13)
        risk_inputs = []
        hook = self.ex.memory.risk.register_forward_pre_hook(
            lambda module, args: risk_inputs.append(args[0].detach().clone()))
        self.addCleanup(hook.remove)
        prefix = self.ex.rollout(encoded[:6], "adaptive", record_audit=True)
        prefix_inputs = list(risk_inputs)
        risk_inputs.clear()
        full = self.ex.rollout(encoded, "adaptive", record_audit=True)
        self.assertEqual(prefix["chain"], full["chain"][:5])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(prefix_inputs, risk_inputs[:5])))
        self.assertAlmostEqual(float(risk_inputs[-1][-1]), math.log1p(8), places=6)
        self.assertEqual([d["keep_streak_before"] for d in full["decisions"]],
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8])
        self.assertEqual(full["keep_streak"], 8)
        self.assertTrue(all(d["action"] == 0 for d in full["decisions"]))

    def test_real_shrink_and_expand_reset_streak(self):
        result = self.ex.rollout(stream(5), "beamplan", train=True, prefix_seed=1,
                                 forced_sequence=[0, 0, 1, 2], suffix_start=1)
        self.assertEqual([d["action"] for d in result["decisions"]], [0, 0, 1, 2])
        self.assertEqual([d["keep_streak_before"] for d in result["decisions"]], [0, 1, 2, 0])
        self.assertEqual(result["keep_streak"], 0)

    def test_beam_window_uses_current_policy_outside_intervention(self):
        result = self.ex.rollout(stream(6), "beamplan", train=True, prefix_seed=17,
                                 forced_sequence=[1], suffix_start=3)
        self.assertEqual([d["action"] for d in result["decisions"]], [0, 0, 1, 0, 0])
        self.assertEqual(result["decisions"][2]["step"], 3)

    def test_noop_expansion_counts_as_keep_not_reset(self):
        result = self.ex.rollout(stream(3), "beamplan", train=True,
                                 forced_sequence=[0, 2], suffix_start=1)
        self.assertEqual([d["requested_action"] for d in result["decisions"]], [0, 2])
        self.assertEqual([d["action"] for d in result["decisions"]], [0, 0])
        self.assertEqual(result["keep_streak"], 2)
        self.assertEqual(self.ex.actions, dict(keep=2, shrink=0, expand=0))

    def test_minimum_width_shrink_is_noop(self):
        result = self.ex.rollout(stream(2, width=4), "beamplan", train=True,
                                 forced_sequence=[1], suffix_start=1)
        self.assertEqual(result["decisions"][0]["requested_action"], 1)
        self.assertEqual(result["decisions"][0]["action"], 0)
        self.assertEqual(result["keep_streak"], 1)

    def test_beam_and_counterfactual_are_fit_only(self):
        with self.assertRaisesRegex(ValueError, "beam_fit_only"):
            self.ex.rollout(stream(2), "beamplan", train=False,
                            forced_sequence=[1], suffix_start=1)
        with self.assertRaisesRegex(ValueError, "beam_fit_only"):
            self.ex.rollout(stream(2), "beamplan", train=True, suffix_start=1)
        with self.assertRaisesRegex(ValueError, "counterfactual_fit_only"):
            self.ex.rollout(stream(2), "counterfactual", forced=1, last_decision_step=1)

    def test_shrink_expand_uses_retained_state_without_inplace_mutation(self):
        encoded = stream(3)
        original = [event["values"].clone() for event in encoded]
        result = self.ex.rollout(encoded, "beamplan", train=True, record_audit=True,
                                 forced_sequence=[1, 2], suffix_start=1)
        self.assertEqual([d["action"] for d in result["decisions"]], [1, 2])
        self.assertEqual(result["chain"][0]["after_length"], 9)
        self.assertEqual(result["chain"][1]["after_length"], 10)
        self.assertEqual(result["chain"][0]["after_sha"], result["chain"][1]["before_sha"])
        self.assertFalse(torch.equal(result["bank"][0]["values"], original[0]))
        self.assertTrue(all(torch.equal(event["values"], before)
                            for event, before in zip(encoded, original)))
        self.assertTrue(all("body_ids" not in entry for entry in result["bank"]))

    def _mock_search_loss(self, positive):
        encoded = stream(5)
        self.ex.encode_case = lambda case: encoded
        self.ex.forward = lambda pairs, enabled: torch.ones(len(pairs))
        result = {
            "winner_sequence": [2, 1] if positive else [0, 0],
            "winner_realized_sequence": [0, 1] if positive else [0, 0],
            "first_divergence": 1 if positive else None,
            "positive": positive,
            "action_target": 1 if positive else None,
            "evaluated_count": 7,
            "qualified_count": int(positive),
        }
        case = {"group": "fit_fake", "questions": [{"question": "a", "answer": "b"}]}
        with tempfile.TemporaryDirectory() as tmp:
            self.ex.out = Path(tmp)
            with patch.object(run, "beam_search", return_value=result):
                loss, ratio = self.ex.search_loss([case], 0)
            records = [json.loads(line) for line in
                       (self.ex.out / "SEARCH_TEACHER.jsonl").read_text().splitlines()]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(math.isfinite(ratio))
        loss.backward()
        self.assertIsNotNone(self.ex.memory.risk[-1].bias.grad)
        return records[0]

    def test_first_realized_divergence_maps_to_correct_causal_decision(self):
        record = self._mock_search_loss(positive=True)
        self.assertGreaterEqual(record["suffix_start"], 1)
        self.assertLessEqual(record["suffix_start"], 3)
        self.assertEqual(record["supervised_decision_step"], record["suffix_start"] + 1)
        self.assertEqual(record["supervised_action"], 1)
        self.assertGreater(record["keep_penalty_weight"], 0)
        self.assertLessEqual(record["keep_penalty_weight"], 0.02)
        self.assertEqual(self.ex.search_counts["positive"], 1)

    def test_no_positive_teaches_first_suffix_keep_without_keep_penalty(self):
        record = self._mock_search_loss(positive=False)
        self.assertEqual(record["supervised_decision_step"], record["suffix_start"])
        self.assertEqual(record["supervised_action"], 0)
        self.assertEqual(record["keep_penalty_weight"], 0)
        self.assertEqual(self.ex.search_counts["no_positive"], 1)
        self.assertEqual(self.ex.search_counts["keep_penalty_terms"], 0)

    def test_keep_stores_exact_incoming_and_zero_residual_gate(self):
        encoded = stream(4)
        original = [event["values"].clone() for event in encoded]
        result = self.ex.rollout(encoded, "noshrink", record_audit=True)
        self.assertEqual(result["residual_gate"], 0.0)
        self.assertTrue(all(d["action"] == 0 for d in result["decisions"]))
        self.assertTrue(all(torch.equal(entry["values"], before)
                            for entry, before in zip(result["bank"], original)))
        self.assertTrue(all(entry["origin_sha"] == digest_tensor(entry["values"])
                            for entry in result["bank"]))

    def test_shrink_enables_residual_gate_without_rewriting_other_entries(self):
        encoded = stream(3)
        original = [event["values"].clone() for event in encoded]
        result = self.ex.rollout(encoded, "beamplan", train=True, record_audit=True,
                                 forced_sequence=[1], suffix_start=1)
        self.assertEqual(result["residual_gate"], 1.0)
        self.assertEqual(result["decisions"][0]["action"], 1)
        self.assertFalse(torch.equal(result["bank"][0]["values"], original[0]))
        self.assertTrue(torch.equal(result["bank"][1]["values"], original[1]))

    def test_maintenance_and_keep_prefix_create_long_streak_states(self):
        encoded = stream(3)
        self.ex.maintenance = 12
        result = self.ex.rollout(encoded, "beamplan", train=True, forced_sequence=[1],
                                 suffix_start=11, keep_prefix_from=3)
        self.assertEqual(len(result["decisions"]), 14)
        self.assertGreaterEqual(result["decisions"][10]["keep_streak_before"], 8)
        self.assertEqual(result["decisions"][10]["action"], 1)
        self.assertEqual(result["decisions"][11]["keep_streak_before"], 0)

    def test_intervention_covers_late_positions_and_long_streaks(self):
        late_long = run.choose_intervention(draw=8, n_events=4, depth=2, maintenance=12)
        self.assertEqual(late_long["position_bucket"], "late")
        self.assertEqual(late_long["target_keep_streak"], 8)
        self.assertGreaterEqual(late_long["start"], 9)
        self.assertEqual(late_long["keep_prefix_from"], late_long["start"] - 8)

    def test_arm_flags_cross_quota_with_codec_and_slot(self):
        self.assertEqual(run.arm_flags("control"),
                         dict(quota=False, codec=False, slot=False, dynpen=False, curriculum=False))
        self.assertEqual(run.arm_flags("curriculum"),
                         dict(quota=False, codec=True, slot=True, dynpen=False, curriculum=True))
        self.assertTrue(run.arm_flags("dynpen_slot")["dynpen"])
        self.assertAlmostEqual(run.force_unkeep_rate(0.0), 0.75)
        self.assertAlmostEqual(run.force_unkeep_rate(1.0), 0.15)
        self.assertEqual(run.rate_band_penalty(0.05, 0.50)[0], "keep")
        self.assertEqual(run.rate_band_penalty(0.90, 0.22)[0], "unkeep")
        self.assertEqual(run.rate_band_penalty(0.50, 0.50)[0], "none")

    def test_dynamic_keep_penalty_is_zero_at_target_and_strong_when_all_keep(self):
        self.assertAlmostEqual(run.dynamic_keep_penalty(0.5, 0), 0.5 * 0 + 0.3 * (1 / 8))
        self.assertAlmostEqual(run.dynamic_keep_penalty(1.0, 7), 0.8)
        self.assertLess(run.dynamic_keep_penalty(0.4, 0), run.dynamic_keep_penalty(1.0, 7))

    def test_least_harm_shrink_prefers_smaller_harm_then_larger_savings(self):
        records = [
            {"sequence": [0, 0], "realized_sequence": [0, 0], "cost_savings_vs_keep": 0.2, "max_question_harm": 0.0},
            {"sequence": [1, 0], "realized_sequence": [1, 0], "cost_savings_vs_keep": 0.1, "max_question_harm": 0.02},
            {"sequence": [1, 1], "realized_sequence": [1, 1], "cost_savings_vs_keep": 0.3, "max_question_harm": 0.02},
            {"sequence": [1, 0], "realized_sequence": [0, 0], "cost_savings_vs_keep": 0.4, "max_question_harm": 0.0},
        ]
        picked = run.least_harm_shrink(records)
        self.assertEqual(picked["sequence"], [1, 1])
        self.assertEqual(picked["action"], 1)
        self.assertEqual(picked["divergence"], 0)
        self.assertIsNone(run.least_harm_shrink(records, max_harm=0.01))
        self.assertEqual(run.least_harm_shrink(records, max_harm=0.02)["sequence"], [1, 1])
