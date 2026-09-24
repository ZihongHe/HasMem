"""Checks for the published local judging protocol and coverage guards."""
import json
import tempfile
import unittest
from pathlib import Path
from hasmem.judge import get_anscheck_prompt, load_inputs, parse_yes


class JudgeTests(unittest.TestCase):
    def fixtures(self, root, count=500):
        refs = [{"question_id": str(i), "question_type": "multi-session", "question": "Q", "answer": "A"} for i in range(count)]
        rows = [{"dataset": "longmemeval_s", "case_id": f"lme_s:{i}/0", "condition": mode, "prediction": "A"}
                for mode in ("hard", "adaptive") for i in range(count)]
        rp, hp = root / "refs.json", root / "rows.jsonl"
        rp.write_text(json.dumps(refs), encoding="utf-8")
        hp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return rp, hp, rows

    def test_full_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            rp, hp, _ = self.fixtures(Path(directory))
            refs, groups = load_inputs(hp, rp)
            self.assertEqual(len(refs), 500)
            self.assertEqual([len(v) for v in groups.values()], [500, 500])

    def test_missing_and_duplicate_predictions_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            rp, hp, rows = self.fixtures(Path(directory))
            for bad, message in ((rows[:-1], "Incomplete"), (rows + rows[:1], "Duplicate")):
                hp.write_text("\n".join(json.dumps(r) for r in bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_inputs(hp, rp)

    def test_subset_requires_explicit_flag_and_matching_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            rp, hp, rows = self.fixtures(Path(directory), 2)
            with self.assertRaisesRegex(ValueError, "500"):
                load_inputs(hp, rp)
            self.assertEqual(len(load_inputs(hp, rp, True)[0]), 2)
            hp.write_text("\n".join(json.dumps(r) for r in rows[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical"):
                load_inputs(hp, rp, True)

    def test_task_prompts_preserve_distinct_rules(self):
        self.assertIn("off-by-one", get_anscheck_prompt("temporal-reasoning", "Q", "A", "R"))
        self.assertIn("updated answer", get_anscheck_prompt("knowledge-update", "Q", "A", "R"))
        self.assertIn("Rubric: A", get_anscheck_prompt("single-session-preference", "Q", "A", "R"))
        self.assertIn("unanswerable", get_anscheck_prompt("multi-session", "Q", "A", "R", True))
        with self.assertRaises(NotImplementedError):
            get_anscheck_prompt("unknown", "Q", "A", "R")

    def test_historical_label_parser(self):
        for text in ("yes", "Yes.", "Correct", "The answer is yes"):
            self.assertTrue(parse_yes(text))
        for text in ("no", "No, yes was wrong", "incorrect", "wrong", ""):
            self.assertFalse(parse_yes(text))
