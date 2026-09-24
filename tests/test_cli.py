"""Preflight validation and reproducible model-revision loading."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hasmem.cli import load_config, main, make_plan, validate_inputs
from hasmem.compatibility import inference_metadata, validate_checkpoint_metadata


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        prepared = self.root / 'prepared'
        prepared.mkdir()
        for split in ('train', 'validation'):
            (prepared / (split + '.jsonl')).write_text('', encoding='utf-8')
        (prepared / 'manifest.json').write_text(json.dumps({'trace_paths': {
            'train': 'train.jsonl', 'validation': 'validation.jsonl'}}), encoding='utf-8')
        self.config = {'model': 'Qwen/Qwen2.5-7B-Instruct',
                       'data': {'manifest': 'prepared/manifest.json'}}
        self.checkpoint = self.root / 'checkpoint.pt'
        self.checkpoint.write_bytes(b'preflight checks existence only')
        self.records = self.root / 'records.json'
        self.records.write_text('["A constructed record."]', encoding='utf-8')

    def test_relative_manifest_paths_and_no_early_holdout_read(self):
        holdout = self.root / 'holdout.json'
        holdout.write_text('not read by preflight', encoding='utf-8')
        self.config['data']['longmemeval'] = 'holdout.json'
        self.assertEqual(validate_inputs(self.config, self.root, 'train'), {})
        holdout.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'data.longmemeval'):
            validate_inputs(self.config, self.root, 'train')

    def test_missing_manifest_and_each_trace_fail(self):
        for split in ('train', 'validation'):
            trace = self.root / 'prepared' / (split + '.jsonl')
            trace.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'trace_paths.' + split):
                validate_inputs(self.config, self.root, 'train')
            trace.write_text('', encoding='utf-8')
        (self.root / 'prepared' / 'manifest.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'data.manifest'):
            validate_inputs(self.config, self.root, 'train')

    def test_missing_trace_mapping_fails(self):
        (self.root / 'prepared' / 'manifest.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'trace_paths'):
            validate_inputs(self.config, self.root, 'train')

    def test_evaluation_requires_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, '--checkpoint'):
            validate_inputs(self.config, self.root, 'evaluate', self.root / 'missing.pt')
        checked = validate_inputs(self.config, self.root, 'evaluate', self.checkpoint)
        self.assertEqual(checked['checkpoint'], self.checkpoint)

    def test_inference_requires_no_msc_data(self):
        checked = validate_inputs({'model': 'example'}, self.root, 'infer', self.checkpoint, self.records)
        self.assertEqual(checked['records'], ['A constructed record.'])

    def test_inference_rejects_missing_and_invalid_records(self):
        with self.assertRaisesRegex(FileNotFoundError, '--records'):
            validate_inputs(self.config, self.root, 'infer', self.checkpoint, self.root / 'missing.json')
        for invalid in ('[]', '{}', '[""]', '[2]'):
            self.records.write_text(invalid, encoding='utf-8')
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'records must'):
                validate_inputs(self.config, self.root, 'infer', self.checkpoint, self.records)

    def test_invalid_input_stops_before_model_loading_or_output_creation(self):
        cfg = self.root / 'config.json'
        self.config['data']['manifest'] = 'missing.json'
        cfg.write_text(json.dumps(self.config), encoding='utf-8')
        output = self.root / 'output'
        with patch('sys.argv', ['hasmem', 'train', '--config', str(cfg), '--output', str(output)]), \
                patch('hasmem.core.AutoTokenizer.from_pretrained') as tokenizer:
            with self.assertRaisesRegex(FileNotFoundError, 'data.manifest'):
                main()
            tokenizer.assert_not_called()
        self.assertFalse(output.exists())

    def test_check_config_checks_inputs_without_model_loading(self):
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps(self.config), encoding='utf-8')
        stream = io.StringIO()
        with patch('sys.argv', ['hasmem', 'check-config', '--config', str(cfg)]), \
                patch('hasmem.core.AutoTokenizer.from_pretrained') as tokenizer, contextlib.redirect_stdout(stream):
            main()
            tokenizer.assert_not_called()
        self.assertTrue(json.loads(stream.getvalue())['valid'])
        (self.root / 'prepared' / 'train.jsonl').unlink()
        with patch('sys.argv', ['hasmem', 'check-config', '--config', str(cfg)]):
            with self.assertRaisesRegex(FileNotFoundError, 'trace_paths.train'):
                main()

    def test_optional_revision_is_validated_and_forwarded(self):
        cfg = self.root / 'config.json'
        for invalid in ('', '   ', 123):
            cfg.write_text(json.dumps({**self.config, 'model_revision': invalid}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'model_revision'):
                load_config(cfg)
        _, spec, _ = make_plan(self.config, self.root, 'train')
        self.assertNotIn('model_revision', spec)
        _, spec, _ = make_plan({**self.config, 'model_revision': 'snapshot-sha'}, self.root, 'train')
        self.assertEqual(spec['model_revision'], 'snapshot-sha')

    def test_revision_reaches_both_loaders_and_omission_preserves_defaults(self):
        from hasmem.core import Experiment
        for revision in (None, 'snapshot-sha'):
            spec = {'model': 'example-model', 'seed': 13}
            if revision is not None:
                spec['model_revision'] = revision
            with patch('hasmem.core.AutoTokenizer.from_pretrained') as tokenizer, \
                    patch('hasmem.core.AutoModelForCausalLM.from_pretrained', side_effect=RuntimeError('stop before allocation')) as model:
                with self.assertRaisesRegex(RuntimeError, 'stop before allocation'):
                    Experiment({'local_files_only': True}, spec, self.root)
                for loader in (tokenizer, model):
                    self.assertEqual(loader.call_args.args, ('example-model',))
                    self.assertTrue(loader.call_args.kwargs['local_files_only'])
                    if revision is None:
                        self.assertNotIn('revision', loader.call_args.kwargs)
                    else:
                        self.assertEqual(loader.call_args.kwargs['revision'], revision)

    def test_revision_pin_preserves_existing_checkpoint_metadata(self):
        def experiment(pin):
            return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash='resolved-sha')),
                tok=SimpleNamespace(get_vocab=lambda: {'x': 0}, chat_template='template', special_tokens_map={}),
                spec={'model': 'example-model', **({'model_revision': pin} if pin else {})},
                seed=13, backbone_tag='7B', prompt_slots=8, maintenance=6, flags={})
        old = inference_metadata(experiment(None))
        current = inference_metadata(experiment('resolved-sha'))
        self.assertEqual(validate_checkpoint_metadata(old, current), 'verified')
