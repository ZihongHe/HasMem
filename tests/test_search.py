"""Pure CPU tests for the bounded counterfactual search contract."""

import math
import unittest

from hasmem.search import beam_search


def record(nll=(1.0, 1.0), cost=1.0, penalty=0.0):
    return {
        "per_question_nll": list(nll),
        "cumulative_cost": cost,
        "keep_streak_penalty": penalty,
    }


class SearchTest(unittest.TestCase):
    def test_unique_best_safe_target(self):
        values = {
            (0,): record(cost=1.0, penalty=0.01),
            (1,): record(cost=0.9),
            (2,): record(cost=0.8),
        }
        result = beam_search(1, values.__getitem__)
        self.assertTrue(result["positive"])
        self.assertEqual(result["winner_sequence"], [1])
        self.assertEqual(result["action_target"], 1)
        self.assertEqual(result["first_divergence"], 0)
        self.assertGreater(result["advantage"], 0)
        self.assertGreater(result["base_advantage"], 0)

    def test_no_forced_positive_when_all_compressions_harmful(self):
        def score(path):
            return record(nll=(1.0, 1.0), penalty=100.0) if path == (0, 0) else record(
                nll=(0.1, 1.001), cost=0.1
            )

        result = beam_search(2, score)
        self.assertFalse(result["positive"])
        self.assertEqual(result["winner_sequence"], [0, 0])
        self.assertIsNone(result["action_target"])
        self.assertIsNone(result["first_divergence"])
        self.assertAlmostEqual(result["records"][0]["keep_streak_penalty"], 0.02)

    def test_per_question_harm_cannot_cancel(self):
        values = {
            (0,): record(nll=(1.0, 1.0)),
            (1,): record(nll=(0.5, 1.1), cost=0.5),
            (2,): record(nll=(1.0, 1.0), cost=0.9),
        }
        result = beam_search(1, values.__getitem__)
        self.assertFalse(result["positive"])
        self.assertEqual(result["winner_sequence"], [0])
        harmful = next(r for r in result["records"] if r["sequence"] == [1])
        self.assertFalse(harmful["positive_eligible"])
        self.assertGreater(harmful["max_question_harm"], 0)

    def test_no_positive_without_strict_cost_savings(self):
        result = beam_search(1, lambda path: record(nll=(1.0 - path[0] * 0.1, 1.0), cost=1.0))
        self.assertFalse(result["positive"])

    def test_no_positive_without_objective_improvement(self):
        result = beam_search(
            1,
            lambda path: record(nll=(1.0 + path[0] * 0.00003, 1.0), cost=1.0 - path[0] * 1e-7),
        )
        self.assertFalse(result["positive"])

    def test_deterministic_ties_and_cache_bound(self):
        calls = []

        def score(path):
            calls.append(path)
            return record(cost=1.0 if path == (0, 0) else 0.8)

        result = beam_search(2, score)
        self.assertEqual(result["winner_sequence"], [1, 0])
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(calls), 7)
        self.assertEqual(result["evaluated_count"], len(calls))
        self.assertTrue(all(len(path) == 2 for path in calls))
        second = beam_search(2, lambda path: record(cost=1.0 if path == (0, 0) else 0.8))
        self.assertEqual(result, second)

    def test_first_divergence_is_not_always_first_decision(self):
        def score(path):
            if path == (0, 1):
                return record(nll=(0.9, 0.9), cost=0.8)
            if path == (0, 0):
                return record()
            return record(nll=(1.01, 1.01), cost=0.9)

        result = beam_search(2, score)
        self.assertEqual(result["winner_sequence"], [0, 1])
        self.assertEqual(result["first_divergence"], 1)
        self.assertEqual(result["action_target"], 1)

    def test_expansion_only_cannot_be_positive_even_if_callback_reports_lower_cost(self):
        values = {(0,): record(), (1,): record(nll=(1.1, 1.1), cost=0.9),
                  (2,): record(nll=(0.5, 0.5), cost=0.8)}
        result = beam_search(1, values.__getitem__)
        self.assertFalse(result["positive"])
        self.assertEqual(result["best_nonkeep_sequence"], [2])

    def test_shrink_then_expand_can_qualify_with_net_cumulative_savings(self):
        def score(path):
            if path == (1, 2):
                return record(nll=(0.9, 0.9), cost=0.95)
            if path == (0, 0):
                return record()
            if path == (1, 0):
                return record(cost=0.9)
            return record(nll=(1.1, 1.1), cost=1.1)

        result = beam_search(2, score)
        self.assertTrue(result["positive"])
        self.assertEqual(result["winner_sequence"], [1, 2])
        self.assertEqual(result["action_target"], 1)

    def test_explicit_teacher_used_for_quality_hinge(self):
        result = beam_search(1, lambda path: record(nll=(1.0, 2.0), cost=1.0 - path[0] * 0.1),
                             hard_teacher_nll=[0.5, 2.0])
        self.assertAlmostEqual(result["records"][0]["quality_objective"], 2.5)

    def test_requested_expansion_noop_targets_later_realized_shrink(self):
        def score(path):
            if path == (2, 1):
                return dict(record(nll=(0.9, 0.9), cost=0.8), realized_sequence=[0, 1])
            if path == (0, 0):
                return record()
            if path == (2, 0):
                return dict(record(nll=(1.0, 1.0), cost=1.0), realized_sequence=[0, 0])
            return record(nll=(1.2, 1.2), cost=0.9)

        result = beam_search(2, score)
        self.assertTrue(result["positive"])
        self.assertEqual(result["winner_sequence"], [2, 1])
        self.assertEqual(result["winner_realized_sequence"], [0, 1])
        self.assertEqual(result["first_divergence"], 1)
        self.assertEqual(result["action_target"], 1)

    def test_requested_shrink_noop_cannot_be_positive(self):
        def score(path):
            return dict(record(nll=(1.0 - path[0] * 0.1, 1.0), cost=1.0 - path[0] * 0.1),
                        realized_sequence=[0])

        result = beam_search(1, score)
        self.assertFalse(result["positive"])
        self.assertIsNone(result["first_divergence"])
        self.assertIsNone(result["action_target"])

    def test_invalid_realized_sequence_rejected(self):
        for realized in ([0, 0], [], [3], [-1], [1.0], [True]):
            with self.subTest(realized=realized), self.assertRaises(ValueError):
                beam_search(1, lambda path: dict(record(), realized_sequence=realized))

    def test_bad_scores_fail_closed(self):
        invalid = [record(nll=(math.nan, 1)), record(cost=math.inf), record(cost=-1),
                   record(penalty=-1), record(nll=()), record(nll=(-1, 1))]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                beam_search(1, lambda path, value=value: value)
        with self.assertRaises(ValueError):
            beam_search(1, lambda path: record(nll=(1,) if path == (0,) else (1, 2)))
        with self.assertRaises(ValueError):
            beam_search(1, lambda path: record(), hard_teacher_nll=[1])

    def test_bad_search_configuration(self):
        for depth in (0, 3, 1.0, True):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                beam_search(depth, lambda path: record())
        for width in (0, 3, 1.0, True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                beam_search(1, lambda path: record(), beam_width=width)
        for kw in ({"quality_tolerance": -1}, {"cost_weight": -1}, {"harm_weight": -1},
                   {"keep_penalty_cap": -1}, {"quality_tolerance": math.nan}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                beam_search(1, lambda path: record(), **kw)


if __name__ == "__main__":
    unittest.main()
