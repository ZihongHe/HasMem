"""Derive the actual native assistant-message terminator, not a padding token."""
from __future__ import annotations


def derive_answer_eos(tokenizer) -> int:
    """Return the first special token closing an empty native assistant message.

    The empty completion must extend the exact generation-prompt token prefix.
    We deliberately exclude any subsequent formatting newline from supervision.
    A tokenizer exposes no model generation_config, so consistency is checked
    against its configured eos_token_id; the caller separately owns decode stops.
    """
    messages = [
        {'role': 'system', 'content': 'Use the supplied memory.'},
        {'role': 'user', 'content': 'Return the stored value.'},
    ]
    prefix = list(tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True,
    ))
    complete = list(tokenizer.apply_chat_template(
        messages + [{'role': 'assistant', 'content': ''}],
        tokenize=True, return_dict=False, add_generation_prompt=False,
    ))
    if not prefix or complete[:len(prefix)] != prefix:
        raise ValueError('native_assistant_generation_prefix_mismatch')
    suffix = complete[len(prefix):]
    if not suffix:
        raise ValueError('native_assistant_missing_termination_suffix')
    special = set(tokenizer.all_special_ids)
    terminator = next((tok for tok in suffix if isinstance(tok, int) and tok in special
                       and tok != tokenizer.pad_token_id), None)
    if terminator is None:
        raise ValueError('native_assistant_missing_special_termination')
    return terminator
