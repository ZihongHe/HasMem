"""Data and configuration tests using constructed fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from hasmem import data
from hasmem.cli import load_config
from hasmem.preparation.msc import prepare
from hasmem.metrics import score_prediction


class WhitespaceTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(len(text.split())))}


class DataTests(unittest.TestCase):
    def test_preparation_is_deterministic_and_preserves_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'validation'):
                rows = [{'dialog_id': str(i), 'session_id': 1,
                         'persona1': [f'I collect stamps number {i}.'],
                         'persona2': [f'I live in town number {i}.'],
                         'dialogue': [f'Welcome visitor number {i}.', f'Goodbye visitor number {i}.'],
                         'speaker': ['Speaker 1', 'Speaker 2']} for i in range(4)]
                (root / f'{split}.jsonl').write_text('\n'.join(json.dumps(x) for x in rows), encoding='utf-8')
            first = prepare(root, root / 'prepared', paper_cohort=False)
            before = (first.parent / 'validation.traces.jsonl').read_bytes()
            prepare(root, root / 'prepared', paper_cohort=False)
            self.assertEqual(before, (first.parent / 'validation.traces.jsonl').read_bytes())
            data.configure(first)
            loaded = data.load_data(WhitespaceTokenizer(), validation_owners=4)
            self.assertEqual(len(loaded['validation']), 4)
            self.assertFalse({x['id'] for x in loaded['train']} & {x['id'] for x in loaded['validation']})
            self.assertTrue(all(q['answer'] in '\n'.join(c['events']) for c in loaded['validation'] for q in c['questions']))

    def test_cover_and_brief_differ_for_long_output(self):
        scores = score_prediction('Paris ' + 'extra ' * 20, 'Paris')
        self.assertEqual(scores['cover'], 1)
        self.assertEqual(scores['brief_cover'], 0)
        self.assertEqual(scores['em'], 0)

    def test_missing_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Set model'):
                load_config(path)
