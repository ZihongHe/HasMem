"""Reject checkpoints with incompatible deployment settings before inference."""
import copy
import unittest
from types import SimpleNamespace

from hasmem.compatibility import resolve_backbone_tag, validate_checkpoint_metadata


class CompatibilityTests(unittest.TestCase):
    def test_local_directory_name_does_not_change_known_architecture(self):
        config = SimpleNamespace(model_type='qwen2', hidden_size=3584, num_hidden_layers=28)
        self.assertEqual(resolve_backbone_tag({'model': '/models/download'}, config), '7B')
        self.assertEqual(resolve_backbone_tag({'model': '/models/backup-0.5B'}, config), '7B')

    def test_mistral_local_architecture_resolves(self):
        config = SimpleNamespace(model_type='mistral', hidden_size=4096, num_hidden_layers=32)
        self.assertEqual(resolve_backbone_tag({'model': '/models/local'}, config), '7B')

    def test_unknown_architecture_requires_size_configuration(self):
        with self.assertRaisesRegex(ValueError, 'backbone_tag'):
            resolve_backbone_tag({'model': '/models/local'}, SimpleNamespace())
        self.assertEqual(resolve_backbone_tag({'model': '/models/local', 'backbone_tag': '3B'}, SimpleNamespace()), '3B')

    def test_legacy_checkpoint_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'compatibility metadata'):
            validate_checkpoint_metadata(None, {'version': 1})
        self.assertEqual(validate_checkpoint_metadata(None, {'version': 1}, True), 'legacy_metadata_unverified')

    def test_each_inference_setting_is_checked(self):
        current = {'version': 1, 'engine': 'step', 'model': 'Qwen/Qwen2.5-7B-Instruct',
                   'seed': 13, 'retrieval_k': 8, 'maintenance_steps': 6,
                   'method_flags': {'use_global': True, 'use_soft': True, 'shrink_frac': 0.1},
                   'tokenizer_sha256': 'first'}
        self.assertEqual(validate_checkpoint_metadata(copy.deepcopy(current), current), 'verified')
        for key, value in [('engine', 'wallclock'), ('model', 'another-model'), ('seed', 14),
                           ('retrieval_k', 2), ('maintenance_steps', 12),
                           ('method_flags', {'use_global': False}), ('tokenizer_sha256', 'second')]:
            saved = copy.deepcopy(current)
            saved[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_checkpoint_metadata(saved, current, allow_legacy=True)

    def test_unknown_metadata_version_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            validate_checkpoint_metadata({'version': 2}, {'version': 1})


if __name__ == '__main__':
    unittest.main()
