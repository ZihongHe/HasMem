"""Deterministic MSC-derived and LongMemEval-S data preparation."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path


MANIFEST = None
LME = None
MAX_EVENTS = 16
TRAIN_MAX_EVENTS = 48
MAX_EVENT_TOKENS = 384
MAX_ANSWER_TOKENS = 48
VALIDATION_OWNERS = 268
LME_QUESTIONS = 500
CHUNK_TOKENS = 128


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _tokens(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)['input_ids'])


def _length_bucket(n_events: int) -> str:
    if n_events <= 3:
        return 'short'
    if n_events <= 8:
        return 'medium'
    return 'long'


def _read_msc(path: Path, tokenizer, split: str, max_events: int = MAX_EVENTS):
    counts = Counter()
    cases = []
    owners = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            counts['source_traces'] += 1
            owner = str(raw['owner_id'])
            if owner in owners:
                raise ValueError('MSC source has repeated owner: aggregation protocol required')
            owners.add(owner)
            selected = []
            for step in sorted(raw.get('steps') or [], key=lambda row: int(row.get('step', 0))):
                for event in step.get('events') or []:
                    counts['source_events'] += 1
                    if str(event.get('owner_id', owner)) != owner:
                        raise ValueError('MSC event and trace owners disagree')
                    text = str(event.get('text') or '')
                    label = str(event.get('label') or '')
                    value = str(event.get('value') or '').strip()
                    if not text.strip() or not label.strip() or not value:
                        counts['skipped_empty_event'] += 1
                        continue
                    if len(_tokens(tokenizer, text)) > MAX_EVENT_TOKENS:
                        counts['skipped_event_over_384_tokens'] += 1
                        continue
                    if len(_tokens(tokenizer, value)) > MAX_ANSWER_TOKENS:
                        counts['skipped_value_over_48_tokens'] += 1
                        continue
                    if len(selected) >= max_events:
                        counts['eligible_events_after_cap_not_used'] += 1
                        continue
                    selected.append({'text': text, 'label': label, 'value': value})
            if len(selected) < 2:
                counts['skipped_trace_fewer_than_2_events'] += 1
                continue
            latest = {event['label']: (index, event) for index, event in enumerate(selected)}
            target_labels = list(dict.fromkeys((selected[0]['label'], selected[-1]['label'])))
            questions = []
            for label in target_labels:
                position, event = latest[label]
                questions.append({
                    'question': f'What is the recorded value for {label}? Answer only the value.',
                    'answer': event['value'],
                    'target_label': label,
                    'age': len(selected) - 1 - position,
                })
            cases.append({
                'id': 'msc:' + owner,
                'group': 'msc:' + owner,
                'dataset': 'msc',
                'split': split,
                'events': [event['text'] for event in selected],
                'event_labels': [event['label'] for event in selected],
                'questions': questions,
                'history_events': len(selected),
                'length_bucket': _length_bucket(len(selected)),
                'history_tokens': sum(len(_tokens(tokenizer, event['text'])) for event in selected),
            })
    cases.sort(key=lambda row: _hash(row['group'].removeprefix('msc:')))
    counts['source_owners'] = len(owners)
    counts['eligible_owners'] = len(cases)
    counts['eligible_events'] = sum(len(row['events']) for row in cases)
    counts['eligible_questions'] = sum(len(row['questions']) for row in cases)
    return cases, owners, dict(counts)


def _chronology_key(raw_date, index: int):
    text = str(raw_date or '').strip()
    for fmt in (None, '%Y/%m/%d (%a) %H:%M', '%Y/%m/%d (%A) %H:%M',
                '%Y/%m/%d %H:%M', '%Y/%m/%d', '%Y-%m-%d', '%Y/%m/%d (%a)'):
        try:
            value = datetime.fromisoformat(text.replace('Z', '+00:00')) if fmt is None else datetime.strptime(text, fmt)
            return (0, value.replace(tzinfo=None).isoformat(), index), True
        except ValueError:
            continue
    return (1, '', index), False


def _chunks(tokenizer, text: str) -> list[str]:
    """Character-lossless pieces, each <=128 tokens under this tokenizer.

    Offset boundaries avoid decoding partial UTF-8 token byte sequences. Any
    boundary re-tokenization excess is fixed by shortening that piece only.
    """
    if not text:
        return []
    pieces = []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    char_start, token_start = 0, 0
    while char_start < len(text):
        token_end = min(len(ids), token_start + CHUNK_TOKENS)
        end = len(text) if token_end == len(ids) else int(offsets[token_end - 1][1])
        if end <= char_start:
            raise ValueError('Tokenizer produced no usable character boundary')
        while end > char_start and len(_tokens(tokenizer, text[char_start:end])) > CHUNK_TOKENS:
            end -= 1
        if end <= char_start:
            raise ValueError('Cannot construct a nonempty <=128-token chunk')
        pieces.append(text[char_start:end])
        char_start = end
        while token_start < len(offsets) and int(offsets[token_start][1]) <= char_start:
            token_start += 1
    if ''.join(pieces) != text or any(len(_tokens(tokenizer, piece)) > CHUNK_TOKENS for piece in pieces):
        raise AssertionError('Lossless bounded chunking failed')
    return pieces


def _lme_groups(rows):
    """Cluster any overlapping haystack session IDs, not just identical lists."""
    parent = list(range(len(rows)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    seen = {}
    sessions_by_row = []
    for index, row in enumerate(rows):
        sessions = sorted({str(value) for value in row.get('haystack_session_ids') or []})
        sessions_by_row.append(sessions)
        for session in sessions:
            if session in seen:
                parent[find(index)] = find(seen[session])
            else:
                seen[session] = index
    components = {}
    for index, sessions in enumerate(sessions_by_row):
        components.setdefault(find(index), set()).update(sessions)
    return [
        'lme:' + _hash(json.dumps(sorted(components[find(index)]), separators=(',', ':'))
                       if components[find(index)] else 'empty:' + str(row['question_id']))
        for index, row in enumerate(rows)
    ]


def _read_lme(tokenizer):
    with LME.open(encoding='utf-8') as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or len(raw) != 500:
        raise ValueError('Expected the fixed 500-question official LongMemEval-S source')
    ids = [str(row['question_id']) for row in raw]
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated LongMemEval question IDs')
    groups = _lme_groups(raw)
    selected = sorted(range(len(raw)), key=lambda index: _hash(ids[index]))[:LME_QUESTIONS]
    counts = Counter(source_questions=len(raw), selected_questions=len(selected))
    cases = []
    for index in selected:
        row = raw[index]
        sessions = row.get('haystack_sessions') or []
        dates = row.get('haystack_dates') or []
        if len(dates) != len(sessions):
            raise ValueError('LongMemEval date/session count mismatch')
        dated = [_chronology_key(date, position) for position, date in enumerate(dates)]
        if all(known for _, known in dated):
            order = sorted(range(len(sessions)), key=lambda position: dated[position][0])
        else:
            # Source order is retained consistently if a date cannot be parsed.
            order = list(range(len(sessions)))
            counts['histories_with_unparsed_date_source_order_retained'] += 1
        parts = []
        for position in order:
            session = sessions[position]
            if not isinstance(session, list):
                raise ValueError('LongMemEval session is not a list of turns')
            for turn in session:
                if not isinstance(turn, dict):
                    raise ValueError('LongMemEval turn is not a mapping')
                role = str(turn.get('role') or turn.get('speaker') or 'unknown')
                content = str(turn.get('content') or turn.get('text') or '')
                parts.append(f"[{dates[position]}] {role}: {content}\n")
        history = ''.join(parts)
        if not history:
            raise ValueError('Score-free-selected LongMemEval history is empty')
        events = _chunks(tokenizer, history)
        # The answer is accessed only after score-free selection and memory construction.
        answer = row.get('answer')
        if isinstance(answer, list):
            answer = '; '.join(str(value) for value in answer)
        elif answer is None:
            answer = ''
        else:
            answer = str(answer)
        question_date = str(row.get('question_date') or '')
        question_text = str(row.get('question') or '')
        if question_date:
            question_text = f'Question date: {question_date}\n{question_text}'
        cases.append({
            'id': 'lme_s:' + ids[index],
            'group': groups[index],
            'dataset': 'longmemeval_s',
            'split': 'external_evaluation_only',
            'events': events,
            'questions': [{'question': question_text, 'answer': answer, 'age': None,
                           'question_type': row.get('question_type', ''), 'question_id': ids[index]}],
            'question_date': question_date,
            'history_events': len(events),
            'history_sessions': len(sessions),
            'history_tokens': len(_tokens(tokenizer, history)),
            'event_tokens_total': sum(len(_tokens(tokenizer, event)) for event in events),
            'history_sha256': _hash(history),
        })
    counts['selected_groups'] = len({case['group'] for case in cases})
    counts['selected_history_sessions'] = sum(case['history_sessions'] for case in cases)
    counts['selected_history_chunks'] = sum(len(case['events']) for case in cases)
    counts['selected_history_tokens'] = sum(case['history_tokens'] for case in cases)
    counts['max_selected_history_tokens'] = max(case['history_tokens'] for case in cases)
    return cases, dict(counts)


def configure(manifest=None, longmemeval=None, lme_questions=500):
    """Set user-owned data paths before loading either dataset."""
    global MANIFEST, LME, LME_QUESTIONS
    MANIFEST = Path(manifest).expanduser().resolve() if manifest else None
    LME = Path(longmemeval).expanduser().resolve() if longmemeval else None
    LME_QUESTIONS = int(lme_questions)
    if not 1 <= LME_QUESTIONS <= 500:
        raise ValueError('lme_questions must be between 1 and 500')


def load_lme(tokenizer) -> tuple[list[dict], dict]:
    """Call only after training and checkpoint/method freezing are complete."""
    if LME is None or not LME.is_file():
        raise FileNotFoundError('Set data.longmemeval to the LongMemEval-S JSON file')
    cases, counts = _read_lme(tokenizer)
    return cases, {
        **counts,
        'file': {'path': str(LME), 'sha256': _file_hash(LME)},
        'used_for_training_or_selection': False,
        'history_truncated': False,
        'selection': f'sha256(question_id), first {LME_QUESTIONS}, no answer/evidence filtering',
        'grouping': 'connected components of shared haystack_session_ids across all 500 rows',
        'cohort_sha256': _hash(json.dumps([case['id'] for case in cases], separators=(',', ':'))),
    }


def load_data(tokenizer, include_lme: bool = False, validation_owners: int = VALIDATION_OWNERS) -> dict:
    """Prepare MSC only by default; never open/hash LongMemEval before freeze.

    validation_owners truncates the held-out development owners. The default uses all 268 eligible development owners and 535 questions
    in the paper corpus. The test split stays untouched.
    """
    if MANIFEST is None or not MANIFEST.is_file():
        raise FileNotFoundError('Set data.manifest to the prepared MSC manifest')
    with MANIFEST.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    # Do not open, hash, or inspect manifest test paths.
    paths = {name: (MANIFEST.parent / manifest['trace_paths'][name]).resolve()
             for name in ('train', 'validation')}
    train, train_owners, train_audit = _read_msc(paths['train'], tokenizer, 'train', TRAIN_MAX_EVENTS)
    validation, validation_owner_ids, validation_audit = _read_msc(paths['validation'], tokenizer, 'validation', MAX_EVENTS)
    if train_owners & validation_owner_ids:
        raise ValueError('MSC train/validation owner overlap')
    if len(validation) < validation_owners or not train:
        raise ValueError('Insufficient eligible MSC owners')
    validation = validation[:validation_owners]
    lme, lme_audit = load_lme(tokenizer) if include_lme else ([], {'loaded': False})
    audit = {
        'files': {name: {'path': str(path), 'sha256': _file_hash(path)}
                  for name, path in {'manifest': MANIFEST, **paths}.items()},
        'msc_train': train_audit,
        'msc_validation': {**validation_audit, 'selected_owners': len(validation),
                           'selected_events': sum(len(case['events']) for case in validation),
                           'selected_questions': sum(len(case['questions']) for case in validation)},
        'lme': lme_audit,
        'owner_overlap': 0,
        'train_max_events': TRAIN_MAX_EVENTS,
        'validation_max_events': MAX_EVENTS,
        'train_length_buckets': dict(Counter(case['length_bucket'] for case in train)),
        'msc_test_opened': False,
        'lme_used_for_training_or_selection': False,
        'msc_fields_truncated': False,
        'lme_loaded': bool(include_lme),
        'msc_selection': 'sha256(owner), score-free; first 16 length-eligible events',
        'lme_selection': f'sha256(question_id), first {LME_QUESTIONS}, no answer/evidence filtering',
        'lme_grouping': 'connected components of shared haystack_session_ids across all 500 rows',
        'cohort_sha256': {name: _hash(json.dumps([case['id'] for case in cases], separators=(',', ':')))
                          for name, cases in (('train', train), ('validation', validation), ('lme', lme))},
    }
    return {'train': train, 'validation': validation, 'lme': lme, 'audit': audit}
