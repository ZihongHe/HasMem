"""Integrity and ordering checks for the pinned MSC source download."""
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hasmem.preparation.msc import download_source, MSC_MIRROR_REVISION


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "requires the data extra")
class MSCDownloadTests(unittest.TestCase):
    def fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [{"dialoug_id": 7, "session_id": 2}, {"dialoug_id": 3, "session_id": 0}]
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buffer)
        raw = buffer.getvalue()
        files = {"train": {"path": "data/train.parquet", "sha256": hashlib.sha256(raw).hexdigest(), "rows": 2}}
        return rows, raw, files

    def test_download_preserves_order_and_records_provenance(self):
        rows, raw, files = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            with patch("hasmem.preparation.msc.MSC_MIRROR_FILES", files), patch(
                    "hasmem.preparation.msc.urllib.request.urlopen", return_value=io.BytesIO(raw)) as request:
                source = download_source(Path(directory) / "raw")
            actual = [json.loads(line) for line in (source / "train.jsonl").read_text().splitlines()]
            self.assertEqual(actual, rows)
            self.assertIn(MSC_MIRROR_REVISION, request.call_args.args[0])
            metadata = json.loads((source / "source_metadata.json").read_text())
            self.assertEqual(metadata["splits"]["train"]["export_sha256"],
                             hashlib.sha256((source / "train.jsonl").read_bytes()).hexdigest())

    def test_corrupt_download_does_not_create_output(self):
        _, raw, files = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "raw"
            with patch("hasmem.preparation.msc.MSC_MIRROR_FILES", files), patch(
                    "hasmem.preparation.msc.urllib.request.urlopen", return_value=io.BytesIO(raw + b"changed")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    download_source(target)
            self.assertFalse(target.exists())

    def test_existing_different_source_is_preserved(self):
        _, raw, files = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "train.jsonl"
            target.write_bytes(b"existing input")
            with patch("hasmem.preparation.msc.MSC_MIRROR_FILES", files), patch(
                    "hasmem.preparation.msc.urllib.request.urlopen", return_value=io.BytesIO(raw)):
                with self.assertRaisesRegex(FileExistsError, "differs"):
                    download_source(directory)
            self.assertEqual(target.read_bytes(), b"existing input")
