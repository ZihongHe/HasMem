"""CPU-only tests of native assistant-end derivation; no model dependencies."""
from __future__ import annotations

import unittest

from hasmem.eos import derive_answer_eos


class FakeTokenizer:
    all_special_ids = [100, 101, 102]
    eos_token_id = 101
    pad_token_id = 102

    def __init__(self, prefix=None, completed=None):
        self.prefix = [100, 11, 12] if prefix is None else prefix
        self.completed = self.prefix + [101, 13] if completed is None else completed

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        assert tokenize is True
        if add_generation_prompt:
            assert [row['role'] for row in messages] == ['system', 'user']
            return self.prefix
        assert messages[-1] == {'role': 'assistant', 'content': ''}
        return self.completed


class EOSTests(unittest.TestCase):
    def test_returns_terminator_without_following_newline(self):
        self.assertEqual(derive_answer_eos(FakeTokenizer()), 101)

    def test_refuses_prefix_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'prefix_mismatch'):
            derive_answer_eos(FakeTokenizer(completed=[100, 11, 99, 101, 13]))

    def test_allows_ordinary_suffix_before_special_terminator(self):
        self.assertEqual(derive_answer_eos(FakeTokenizer(completed=[100, 11, 12, 13, 101])), 101)

    def test_refuses_padding_even_if_special_and_configured_as_eos(self):
        tokenizer = FakeTokenizer(completed=[100, 11, 12, 102, 13])
        tokenizer.eos_token_id = 102
        with self.assertRaisesRegex(ValueError, 'missing_special_termination'):
            derive_answer_eos(tokenizer)

    def test_refuses_missing_suffix(self):
        with self.assertRaisesRegex(ValueError, 'missing_termination_suffix'):
            derive_answer_eos(FakeTokenizer(completed=[100, 11, 12]))

    def test_returns_first_nonpadding_special_for_runner_stop_validation(self):
        self.assertEqual(derive_answer_eos(FakeTokenizer(completed=[100, 11, 12, 100, 13])), 100)


if __name__ == '__main__':
    unittest.main()
