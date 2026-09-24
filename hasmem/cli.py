"""Local training, checkpoint evaluation, and record-based inference."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import time
from pathlib import Path


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('engine', 'step') not in {'step', 'wallclock'}:
        raise ValueError('engine must be step or wallclock')
    if not config.get('model'):
        raise ValueError('Set model to a Hugging Face model ID or local model directory')
    if config.get('engine') == 'wallclock' and config.get('method'):
        raise ValueError('The wallclock recipe has fixed method settings; use the step engine for ablations')
    if 'backbone_tag' in config:
        from .compatibility import BACKBONE_TAGS
        if config['backbone_tag'] not in BACKBONE_TAGS:
            raise ValueError('backbone_tag must be one of ' + ', '.join(BACKBONE_TAGS))
    if int(config.get('retrieval_k', 8)) < 1:
        raise ValueError('retrieval_k must be positive')
    if int(config.get('max_updates', 5465)) <= 384:
        raise ValueError('max_updates must include 384 warmup steps and a policy phase')
    return config, path.parent


def local_path(value, root):
    if not value:
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return str((root / path).resolve()) if not path.is_absolute() else str(path.resolve())


def make_plan(config, root, command):
    from . import data
    paths = config.get('data', {})
    data.configure(local_path(paths.get('manifest'), root), local_path(paths.get('longmemeval'), root),
                   paths.get('lme_questions', 500))
    runtime = config.get('runtime', {})
    training_seconds = float(runtime.get('training_seconds', 21600))
    evaluation_seconds = float(runtime.get('evaluation_seconds', 86400))
    if training_seconds <= 0 or evaluation_seconds <= 0:
        raise ValueError('Runtime limits must be positive')
    # The original trainers reserve 1200 seconds inside the training deadline.
    # The portable CLI adds that reserve explicitly; evaluation gets its own limit.
    plan = {'deadline_epoch': time.time() + training_seconds + 1200,
            'max_updates': int(config.get('max_updates', 5465)),
            'policy_updates': int(config.get('policy_updates', 5081)),
            'select_in_band': bool(config.get('select_in_band', False)),
            'local_files_only': bool(config.get('local_files_only', False)),
            'allow_legacy_checkpoint': bool(config.get('allow_legacy_checkpoint', False)),
            'inference_only': command == 'infer',
            'configuration': {'wall_hours': (training_seconds + 1200) / 3600,
                              'validation_owners': int(paths.get('validation_owners', 268)),
                              'skip_lme': not bool(paths.get('longmemeval')),
                              'eval_modes': config.get('eval_modes', ['hard', 'adaptive'])}}
    model = str(config['model'])
    if model.startswith(('.', '/', '~')) or Path(model).is_absolute():
        model = local_path(model, root)
    spec = {'label': 'hasmem', 'model': model, 'seed': int(config.get('seed', 2026091331)),
            'arm': 'curriculum', 'prompt_slots': int(config.get('retrieval_k', 8)),
            **config.get('method', {})}
    if 'backbone_tag' in config:
        spec['backbone_tag'] = config['backbone_tag']
    if spec['arm'] != 'curriculum':
        raise ValueError('Use method flags to select ablations; arm must remain curriculum')
    plan['jobs'] = [spec]
    return plan, spec, evaluation_seconds


def restore(experiment, path):
    import torch
    from .compatibility import inference_metadata, validate_checkpoint_metadata
    state = torch.load(path, map_location='cuda', weights_only=True)
    compatibility = validate_checkpoint_metadata(
        state.get('inference_metadata'), inference_metadata(experiment),
        bool(experiment.plan.get('allow_legacy_checkpoint', False)))
    experiment.memory.load_state_dict(state['memory'], strict=True)
    if len(state['adapters']) != len(experiment.adapters):
        raise ValueError('Checkpoint adapter count differs from the configured model')
    for adapter, weights in zip(experiment.adapters, state['adapters']):
        result = adapter.load_state_dict(weights, strict=False)
        if result.unexpected_keys or any(not key.startswith('base.') for key in result.missing_keys):
            raise ValueError('Checkpoint adapter tensors do not match the configured model')
    if int(state.get('seed', experiment.seed)) != experiment.seed:
        raise ValueError('Configured seed must match checkpoint seed')
    return {**{key: state.get(key) for key in ('seed', 'arm', 'updates', 'protocol')},
            'compatibility': compatibility}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'infer', 'check-config'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', help='New output directory; existing directories are rejected')
    parser.add_argument('--checkpoint', help='Trained HasMem adapter checkpoint')
    parser.add_argument('--records', help='JSON array of record strings for inference')
    parser.add_argument('--question')
    args = parser.parse_args()
    config, root = load_config(args.config)
    if args.command == 'check-config':
        print(json.dumps({'valid': True, 'engine': config.get('engine', 'step'),
                          'retrieval_k': config.get('retrieval_k', 8)}))
        return
    if not args.output:
        parser.error('--output is required')
    if args.command in {'evaluate', 'infer'} and not args.checkpoint:
        parser.error('--checkpoint is required')
    if args.command == 'infer' and (not args.records or not args.question):
        parser.error('infer requires --records and --question')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training and LLM inference require a CUDA GPU; CPU unit tests are available')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The paper implementation uses bfloat16; select a GPU supporting it')
    from .core import write
    plan, spec, eval_seconds = make_plan(config, root, args.command)
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'CONFIG.json', config)
    engine = importlib.import_module('hasmem.engines.' + config.get('engine', 'step'))
    started = time.time()
    try:
        ex = engine.Experiment(plan, spec, out)
        if args.command == 'train':
            ex.identity()
            ex.train()
            checkpoint = out / 'checkpoint.pt'
            write(out / 'CHECKPOINT.json', restore(ex, checkpoint))
        else:
            checkpoint = Path(args.checkpoint).expanduser().resolve()
            write(out / 'CHECKPOINT.json', restore(ex, checkpoint))
        if args.command == 'infer':
            records = json.loads(Path(args.records).read_text(encoding='utf-8'))
            if not isinstance(records, list) or not records or any(not isinstance(x, str) or not x.strip() for x in records):
                raise ValueError('records must be a nonempty JSON array of nonempty strings')
            with torch.no_grad():
                encoded = ex.encode_case({'events': records, 'id': 'local_records'})
                rollout = ex.rollout(encoded, 'adaptive', record_audit=True)
                answer, tokens, stopped = ex.generate(rollout, args.question, True)
            write(out / 'answer.json', {'answer': answer, 'generated_tokens': len(tokens),
                                      'stopped_on_eos': stopped,
                                      'slot_widths': [len(entry['values']) for entry in rollout['bank']]})
            print(answer)
        else:
            if spec.get('use_soft', True) and spec.get('origin', 'hard') == 'hard' and not spec.get('open_residual'):
                ex.keep_invariant()
            ex.plan['deadline_epoch'] = time.time() + eval_seconds + 100
            complete, rows = ex.evaluate()
            write(out / 'COMPLETION.json', {'status': 'complete' if complete else 'partial_timeout',
                                          'rows': rows, 'seconds': time.time() - started})
            if not complete:
                raise RuntimeError('Evaluation timed out; partial results must not be treated as a full benchmark')
    except Exception as error:
        write(out / 'ERROR.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
