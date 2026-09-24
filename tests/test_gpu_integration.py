"""Optional CUDA integration with random tiny weights and constructed data.

No pretrained model, network request, or benchmark score is involved.
"""
import json
import tempfile
import unittest
from pathlib import Path
import torch


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'requires CUDA bfloat16')
class GPUIntegrationTests(unittest.TestCase):
    def test_train_restore_and_generate(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from hasmem import data
        from hasmem.cli import restore
        from hasmem.engines.step import Experiment
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / 'Qwen2.5-0.5B-test'
            model_dir.mkdir()
            vocab = {word: i for i, word in enumerate(['[UNK]', '[PAD]', '[EOS]', '[SYSTEM]', '[USER]', '[ASSISTANT]', 'Memory', 'record', ':', 'one', 'two', 'three', 'four', 'What', 'is', 'the', 'value', '?', '.', 'Answer', 'only'])}
            backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]', additional_special_tokens=['[SYSTEM]', '[USER]', '[ASSISTANT]'])
            tokenizer.chat_template = "{% for message in messages %}{{ '[' + message['role'].upper() + ']' }}{{ message['content'] }}[EOS]{% endfor %}{% if add_generation_prompt %}[ASSISTANT]{% endif %}"
            tokenizer.save_pretrained(model_dir)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024, eos_token_id=vocab['[EOS]'], pad_token_id=vocab['[PAD]']))
            model.save_pretrained(model_dir)
            del model
            for split in ('train', 'validation'):
                cases = [{'owner_id': f'{split}_{i}', 'steps': [{'step': 0, 'events': [
                    {'text': 'one two three four', 'label': 'first', 'value': 'one'},
                    {'text': 'two three four one', 'label': 'last', 'value': 'two'}]}]} for i in range(4)]
                (root / f'{split}.jsonl').write_text('\n'.join(json.dumps(x) for x in cases), encoding='utf-8')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'trace_paths': {x: f'{x}.jsonl' for x in ('train', 'validation')}}), encoding='utf-8')
            data.configure(manifest)
            out = root / 'run';out.mkdir()
            plan = {'deadline_epoch': time.time() + 1800, 'smoke_only': True, 'local_files_only': True,
                    'configuration': {'validation_owners': 4, 'skip_lme': True, 'eval_modes': ['hard', 'adaptive']}}
            spec = {'label': 'tiny_test', 'model': str(model_dir), 'seed': 13, 'arm': 'curriculum', 'prompt_slots': 2}
            ex = Experiment(plan, spec, out)
            # The smoke run shortens training but retains deployment maintenance.
            ex.maintenance = 12
            ex.identity()
            ex.train()
            restore(ex, out / 'checkpoint.pt')
            ex.keep_invariant()
            encoded = ex.encode_case(ex.data['validation'][0])
            with torch.no_grad():
                rollout = ex.rollout(encoded, 'adaptive')
                answer, tokens, stopped = ex.generate(rollout, 'What is the value?')
            self.assertIsInstance(answer, str)
            self.assertTrue(0 < len(tokens) <= 64)
            self.assertTrue((out / 'TRAINING.json').is_file())
            complete, row_count = ex.evaluate()
            self.assertTrue(complete)
            self.assertEqual(row_count, 16)
            summary = json.loads((out / 'SUMMARY.json').read_text(encoding='utf-8'))
            self.assertIn('msc', summary['question_weighted_metrics'])
            # Exercise the public inference CLI without dataset paths.
            from unittest.mock import patch
            from hasmem.cli import main
            config = root / 'local.json'
            config.write_text(json.dumps({'model': str(model_dir), 'engine': 'step',
                                          'seed': 13, 'retrieval_k': 2, 'local_files_only': True}), encoding='utf-8')
            records = root / 'records.json'
            records.write_text(json.dumps(['one two three four', 'two three four one']), encoding='utf-8')
            del ex
            torch.cuda.empty_cache()
            from contextlib import redirect_stdout
            import io
            with redirect_stdout(io.StringIO()), patch('sys.argv', ['hasmem', 'infer', '--config', str(config), '--checkpoint',
                                    str(out / 'checkpoint.pt'), '--records', str(records),
                                    '--question', 'What is the value?', '--output', str(root / 'inference')]):
                main()
            self.assertTrue((root / 'inference' / 'answer.json').is_file())
