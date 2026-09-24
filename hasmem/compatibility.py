"""Backbone and checkpoint identity checks for reproducible inference."""
from __future__ import annotations

import hashlib
import json
import re

BACKBONE_TAGS = ('14B', '13B', '8B', '7B', '3B', '1.5B', '0.5B')
LARGE_BACKBONES = {'3B', '7B', '8B', '13B', '14B'}


def named_backbone_tag(value):
    name = str(value).replace(chr(92), '/').rstrip('/').split('/')[-1]
    for tag in BACKBONE_TAGS:
        if re.search(r'(?<![0-9.])' + re.escape(tag) + r'(?![0-9])', name, re.IGNORECASE):
            return tag
    return 'unknown'


def resolve_backbone_tag(spec, model_config):
    """Resolve model size without relying on a local directory's name."""
    explicit = spec.get('backbone_tag')
    if explicit is not None:
        if explicit not in BACKBONE_TAGS:
            raise ValueError('backbone_tag must be one of ' + ', '.join(BACKBONE_TAGS))
        return explicit
    model_type = getattr(model_config, 'model_type', '')
    shape = (getattr(model_config, 'hidden_size', None), getattr(model_config, 'num_hidden_layers', None))
    if model_type == 'qwen2':
        known = {(896, 24): '0.5B', (1536, 28): '1.5B', (2048, 36): '3B',
                 (3584, 28): '7B', (5120, 48): '14B'}
        if shape in known:
            return known[shape]
    if model_type == 'mistral' and shape == (4096, 32):
        return '7B'
    for value in (spec.get('model', ''), getattr(model_config, '_name_or_path', '')):
        tag = named_backbone_tag(value)
        if tag != 'unknown':
            return tag
    raise ValueError('Cannot identify backbone size; set backbone_tag explicitly in the configuration')


def inference_metadata(experiment):
    """Cache the configuration and tokenizer identity alongside adapter weights."""
    if not hasattr(experiment, '_inference_metadata'):
        config = experiment.model.config
        fields = ('model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                  'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'tie_word_embeddings')
        tokenizer = {'vocabulary': experiment.tok.get_vocab(),
                     'chat_template': experiment.tok.chat_template,
                     'special_tokens_map': experiment.tok.special_tokens_map}
        token_hash = hashlib.sha256(json.dumps(tokenizer, sort_keys=True, ensure_ascii=False,
                                                separators=(',', ':')).encode('utf-8')).hexdigest()
        experiment._inference_metadata = {
            'version': 1,
            'engine': type(experiment).__module__.rsplit('.', 1)[-1],
            'model': str(experiment.spec['model']),
            'model_revision': getattr(config, '_commit_hash', None),
            'architecture': {key: getattr(config, key, None) for key in fields},
            'tokenizer_sha256': token_hash,
            'seed': int(experiment.seed),
            'backbone_tag': experiment.backbone_tag,
            'retrieval_k': int(experiment.prompt_slots),
            'maintenance_steps': int(experiment.maintenance),
            'method_flags': dict(getattr(experiment, 'flags', {})),
        }
    return experiment._inference_metadata


def validate_checkpoint_metadata(saved, current, allow_legacy=False):
    if saved is None:
        if not allow_legacy:
            raise ValueError('Checkpoint has no compatibility metadata; use a release checkpoint, or explicitly set allow_legacy_checkpoint after verifying its original configuration')
        return 'legacy_metadata_unverified'
    if not isinstance(saved, dict) or saved.get('version') != 1:
        raise ValueError('Unsupported checkpoint compatibility metadata')
    mismatches = [key for key in current if saved.get(key) != current[key]]
    if mismatches:
        raise ValueError('Checkpoint/configuration mismatch: ' + ', '.join(mismatches))
    return 'verified'
