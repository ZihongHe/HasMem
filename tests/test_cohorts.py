"""Fail-closed reconstruction from public, text-free cohort membership."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from hasmem.preparation.msc import atomic_record, canonical_traces, import_traces, reconstruct_cohort, trace_digest


class CohortTests(unittest.TestCase):
    def fixture(self):
        fact = {"record_id": "fact:example", "question_id": "example", "owner_id": "owner",
                "task": "global", "label": "P1-0", "answer": "I collect stamps."}
        event = atomic_record(fact, "G")
        raw = [{"owner_id": "owner", "steps": [{"step": 0, "events": [event]}]}]
        clean = canonical_traces(raw)
        digest = hashlib.sha256(json.dumps(clean[0]["steps"][0]["events"][0], ensure_ascii=False,
                                           sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cohort = {"model_inputs_sha256": trace_digest(raw), "traces": [{"owner_id": "owner", "steps": [
            {"step": 0, "events": [{"record_id": "fact:example", "sha256": digest}]}]}]}
        return fact, raw, clean, cohort

    def test_exact_reconstruction(self):
        fact, _, clean, cohort = self.fixture()
        self.assertEqual(reconstruct_cohort([fact], cohort), clean)

    def test_source_record_failures(self):
        fact, _, _, cohort = self.fixture()
        with self.assertRaisesRegex(ValueError, "Missing"):
            reconstruct_cohort([], cohort)
        changed = dict(fact, answer="Changed")
        with self.assertRaisesRegex(ValueError, "record hash"):
            reconstruct_cohort([changed], cohort)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reconstruct_cohort([fact, fact], cohort)

    def test_order_and_owner_are_verified(self):
        fact, _, _, cohort = self.fixture()
        changed = copy.deepcopy(cohort)
        changed["traces"][0]["owner_id"] = "wrong-owner"
        with self.assertRaisesRegex(ValueError, "model-input hash"):
            reconstruct_cohort([fact], changed)

    def test_trace_import_and_paper_source_guard(self):
        _, raw, clean, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("TRAIN", "VALIDATION"):
                (root / f"GLOBAL_PROMOTION_TRACE_{split}.jsonl").write_text(json.dumps(raw[0]) + "\n", encoding="utf-8")
            manifest = import_traces(root, root / "output")
            out = json.loads((manifest.parent / "train.traces.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(out, clean[0])
            with self.assertRaisesRegex(ValueError, "differs"):
                import_traces(root, root / "wrong", verify_paper=True)
