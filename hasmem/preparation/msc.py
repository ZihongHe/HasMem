"""Deterministic MSC-derived record preparation and verified trace import.

Input: split-specific JSONL with persona1, persona2, dialogue, speaker,
dialog_id (or dialoug_id), and session_id fields.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

def clip_text(value: object, chars: int = 7000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:chars]


def clip_answer(value: object, chars: int = 420) -> str:
    return clip_text(value, chars)


def question_row(
    qid: str,
    question: str,
    answer: object,
    task: str,
    needs_global: bool,
    evidence: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    answer_text = clip_answer(answer)
    question_text = clip_text(question, 700)
    if not answer_text or not question_text:
        return None
    return {
        "question_id": qid,
        "question": question_text,
        "answer": answer_text,
        "task": task,
        "needs_global": bool(needs_global),
        "evidence": list(evidence or []),
    }


def make_msc_episode(item: dict, split: str, index: int) -> Optional[dict]:
    p1 = [clip_answer(x, 220) for x in item.get("persona1", []) if clip_answer(x, 220)]
    p2 = [clip_answer(x, 220) for x in item.get("persona2", []) if clip_answer(x, 220)]
    dialogue = [clip_answer(x, 260) for x in item.get("dialogue", []) if clip_answer(x, 260)]
    speakers = [str(x) for x in item.get("speaker", [])]
    if not (p1 or p2) or len(dialogue) < 2:
        return None
    owner = f"msc:{split}:{item.get('dialog_id', item.get('dialoug_id', index))}"
    episode_id = f"{owner}:session:{item.get('session_id', index)}:{index}"
    global_lines = ["Speaker 1 profile:"] + [f"P1-{i}: {v}" for i, v in enumerate(p1)]
    global_lines += ["Speaker 2 profile:"] + [f"P2-{i}: {v}" for i, v in enumerate(p2)]
    dynamic_lines = ["Dialogue turns:"]
    for i, utterance in enumerate(dialogue):
        speaker = speakers[i] if i < len(speakers) else f"Speaker {(i % 2) + 1}"
        dynamic_lines.append(f"T{i}: {speaker}: {utterance}")
    global_text = clip_text("\n".join(global_lines))
    dynamic_text = clip_text("\n".join(dynamic_lines))
    questions: List[dict] = []
    for speaker, values in ((1, p1[:3]), (2, p2[:3])):
        for slot, answer in enumerate(values):
            row = question_row(
                f"{episode_id}:g:{speaker}:{slot}",
                f"Repeat profile fact P{speaker}-{slot} for Speaker {speaker} exactly.",
                answer,
                "global",
                True,
                [f"P{speaker}-{slot}"],
            )
            if row:
                questions.append(row)
    turn_indices = sorted(set([0, len(dialogue) // 2, len(dialogue) - 1]))
    for turn in turn_indices:
        row = question_row(
            f"{episode_id}:d:{turn}",
            f"Repeat the content of dialogue turn T{turn} exactly, without the speaker label.",
            dialogue[turn],
            "dynamic",
            False,
            [f"T{turn}"],
        )
        if row:
            questions.append(row)
    if p1 and dialogue:
        turn = len(dialogue) - 1
        mixed_answer = f"{p1[0]} || {dialogue[turn]}"
        row = question_row(
            f"{episode_id}:m:0:{turn}",
            f"Return P1-0, then dialogue turn T{turn}, separated by ||.",
            mixed_answer,
            "mixed",
            True,
            ["P1-0", f"T{turn}"],
        )
        if row:
            questions.append(row)
    if len(questions) < 3:
        return None
    return {
        "episode_id": episode_id,
        "owner_id": owner,
        "dataset": "msc",
        "split": split,
        "global_text": global_text,
        "dynamic_text": dynamic_text,
        "full_text": clip_text(global_text + "\n" + dynamic_text, 12000),
        "questions": questions,
    }


def build_msc(split: str, path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            episode = make_msc_episode(json.loads(line), split, index)
            if episode:
                yield episode


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:15], 16)


def normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(text.split())


def load_jsonl(path: str, limit: int = 0) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def episode_id(row: dict) -> str:
    qid = str(row["question_id"])
    marker = ":g:" if row["task"] == "global" else ":d:"
    if marker not in qid:
        raise ValueError(f"cannot recover episode id from {qid}")
    return qid.split(marker, 1)[0]


def session_number(value: str) -> int:
    match = re.search(r":session:([^:]+):", value)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return stable_int(match.group(1)) % 1_000_000


def atomic_record(row: dict, kind: str) -> dict:
    label = str(row["label"])
    answer = str(row["answer"]).strip()
    owner = str(row["owner_id"])
    prefix = "Stable candidate" if kind == "G" else "Dynamic candidate"
    return {
        "record_id": str(row["record_id"]),
        "source_question_id": str(row["question_id"]),
        "owner_id": owner,
        "kind": kind,
        "label": label,
        "value": answer,
        "text": f"Memory owner: {owner}\n{prefix} {label}: {answer}",
        "policy_text": f"Memory record: {answer}",
        "original_question": str(row.get("question", "")),
        "promotion_label": 1 if kind == "G" else 0,
        "label_source": "msc.persona_field" if kind == "G" else "msc.dialogue_field",
    }


def make_pair(split: str, owner: str, source_episode: str, g: dict, d: dict, index: int) -> dict:
    g_record = atomic_record(g, "G")
    d_record = atomic_record(d, "D")
    pair_id = f"promotion:{split}:{hashlib.sha256((g_record['record_id'] + '|' + d_record['record_id']).encode()).hexdigest()[:20]}"
    mixed_question = (
        f"For memory owner {owner}, return stable field {g_record['label']}, then dynamic "
        f"field {d_record['label']}, separated by ||. Return only those two values."
    )
    return {
        "pair_id": pair_id,
        "split": split,
        "owner_id": owner,
        "source_episode_id": source_episode,
        "session_id": session_number(source_episode),
        "pair_index": index,
        "g": g_record,
        "d": d_record,
        "full_hard_text": g_record["text"] + "\n" + d_record["text"],
        "questions": {
            "g": {
                "question_id": f"{pair_id}:G",
                "question": g_record["original_question"] or (
                    f"For memory owner {owner}, repeat stable field {g_record['label']} exactly."
                ),
                "answer": g_record["value"],
            },
            "d": {
                "question_id": f"{pair_id}:D",
                "question": d_record["original_question"] or (
                    f"For memory owner {owner}, repeat dynamic field {d_record['label']} exactly."
                ),
                "answer": d_record["value"],
            },
            "mixed": {
                "question_id": f"{pair_id}:MIXED",
                "question": mixed_question,
                "answer": f"{g_record['value']} || {d_record['value']}",
            },
        },
        "supervision": {
            "scope": "controlled_field_type",
            "g_is_stable_candidate_not_immutability_claim": True,
            "uses_qa_evidence_for_promotion": False,
        },
    }


def clean_component_pair(g: dict, d: dict) -> bool:
    g_value = normalize(g.get("answer", ""))
    d_value = normalize(d.get("answer", ""))
    return bool(g_value and d_value and g_value not in d_value and d_value not in g_value)


def build_pairs(fact_path: str, split: str, pairs_per_episode: int, limit: int = 0) -> List[dict]:
    grouped: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in load_jsonl(fact_path, limit):
        if row.get("task") not in {"global", "dynamic"}:
            continue
        grouped[episode_id(row)][row["task"]].append(row)

    pairs = []
    for source_episode, tasks in sorted(grouped.items(), key=lambda item: stable_int(item[0])):
        globals_ = sorted(tasks["global"], key=lambda row: stable_int(row["record_id"]))
        dynamics = sorted(tasks["dynamic"], key=lambda row: stable_int(row["record_id"]))
        count = min(pairs_per_episode, len(globals_), len(dynamics))
        used_dynamic = set()
        for index in range(count):
            g = globals_[index]
            d = None
            for offset in range(len(dynamics)):
                candidate = dynamics[(index * 2 + 1 + offset) % len(dynamics)]
                if candidate["record_id"] in used_dynamic or not clean_component_pair(g, candidate):
                    continue
                d = candidate
                break
            if d is None:
                continue
            used_dynamic.add(d["record_id"])
            if g["owner_id"] != d["owner_id"]:
                raise ValueError("G/D owner mismatch before control construction")
            pairs.append(make_pair(split, str(g["owner_id"]), source_episode, g, d, index))
    return sorted(pairs, key=lambda row: stable_int(row["pair_id"]))


def make_continuous_traces(rows: Sequence[dict], horizon: int) -> List[dict]:
    by_owner: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_owner[row["owner_id"]].append(row)
    traces = []
    for owner, owner_rows in sorted(by_owner.items()):
        by_session: Dict[int, List[dict]] = defaultdict(list)
        for row in owner_rows:
            by_session[int(row["session_id"])].append(row)
        sessions = sorted(by_session)
        seen: Counter = Counter()
        past_queries: Counter = Counter()
        steps = []
        for position, session in enumerate(sessions):
            pairs = sorted(by_session[session], key=lambda row: stable_int(row["pair_id"]))
            records = {}
            for pair in pairs:
                for kind in ("g", "d"):
                    record = dict(pair[kind])
                    records[record["record_id"]] = record
            for record in records.values():
                key = f"{record['kind']}:{normalize(record['value'])}"
                record["online_features"] = {
                    "arrival_step": position,
                    "past_exact_seen_count": int(seen[key]),
                    "past_query_count": int(past_queries[key]),
                }
                future_sessions = sessions[position + 1 : position + 1 + horizon]
                future_values = []
                for future_session in future_sessions:
                    for future_pair in by_session[future_session]:
                        future_values.extend(
                            f"{future_pair[name]['kind']}:{normalize(future_pair[name]['value'])}"
                            for name in ("g", "d")
                        )
                record["training_only_oracle_future"] = {
                    "horizon_sessions": horizon,
                    "future_exact_reuse_count": future_values.count(key),
                }
            for record in records.values():
                key = f"{record['kind']}:{normalize(record['value'])}"
                seen[key] += 1
            queries = []
            secondary_mixed_queries = []
            for pair in pairs:
                for name in ("g", "d"):
                    query = dict(pair["questions"][name])
                    query.update({
                        "pair_id": pair["pair_id"],
                        "query_type": name.upper(),
                        "requires": [pair[name]["record_id"]],
                    })
                    queries.append(query)
                    key = f"{pair[name]['kind']}:{normalize(pair[name]['value'])}"
                    past_queries[key] += 1
                secondary_mixed_queries.append({
                    **pair["questions"]["mixed"],
                    "pair_id": pair["pair_id"],
                    "query_type": "MIXED_SECONDARY",
                    "requires": [pair["g"]["record_id"], pair["d"]["record_id"]],
                })
            steps.append({
                "step": position,
                "session_id": session,
                "events": list(records.values()),
                "queries": queries,
                "secondary_mixed_queries": secondary_mixed_queries,
            })
        traces.append({
            "trace_id": f"trace:{owner}",
            "owner_id": owner,
            "split": owner_rows[0]["split"],
            "eligible_continuous": len(steps) >= 2,
            "steps": steps,
        })
    return traces


def flatten_msc(path: str, split: str, limit: int) -> List[dict]:
    provisional = []
    for episode in load_jsonl(path):
        if episode.get("dataset") != "msc":
            continue
        for question in episode.get("questions", []):
            if question.get("task") not in {"global", "dynamic"}:
                continue
            evidence = question.get("evidence") or []
            if len(evidence) != 1:
                continue
            owner = episode["owner_id"]
            label = str(evidence[0])
            answer = str(question["answer"]).strip()
            if not answer:
                continue
            provisional.append({
                "record_id": f"fact:{question['question_id']}",
                "question_id": question["question_id"],
                "owner_id": owner,
                "split": split,
                "task": question["task"],
                "label": label,
                "question": f"For memory owner {owner}, {question['question']}",
                "answer": answer,
                "raw_full_text": episode["full_text"],
            })

    provisional.sort(key=lambda row: stable_int(row["record_id"]))
    if limit:
        provisional = provisional[:limit]
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for row in provisional:
        by_task[row["task"]].append(row)

    result = []
    for task, rows in by_task.items():
        if len(rows) < 2:
            continue
        for index, row in enumerate(rows):
            donor = rows[(index + 7919) % len(rows)]
            offset = 1
            while donor["answer"] == row["answer"] or donor["owner_id"] == row["owner_id"]:
                donor = rows[(index + 7919 + offset) % len(rows)]
                offset += 1
                if offset >= len(rows):
                    break
            item = dict(row)
            item["counterfactual_answer"] = donor["answer"]
            item["wrong_owner_id"] = donor["owner_id"]
            result.append(item)
    result.sort(key=lambda row: stable_int(row["record_id"]))
    return result


def prepare(source_dir, output_dir, pairs_per_episode=2, train_fact_limit=0, validation_fact_limit=0, paper_cohort=True):
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    audits = {}
    for split in ('train', 'validation'):
        path = source / f'{split}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(path)
        episodes_path = output / f'{split}.episodes.jsonl'
        facts_path = output / f'{split}.facts.jsonl'
        trace_path = output / f'{split}.traces.jsonl'
        write_jsonl(str(episodes_path), build_msc(split, str(path)))
        limit = 0 if paper_cohort else (train_fact_limit if split == 'train' else validation_fact_limit)
        facts = flatten_msc(str(episodes_path), split, limit)
        write_jsonl(str(facts_path), facts)
        if paper_cohort:
            cohort_path = Path(__file__).with_name('cohorts') / f'{split}.json'
            cohort = json.loads(cohort_path.read_text(encoding='utf-8'))
            traces = reconstruct_cohort(facts, cohort)
        else:
            pairs = build_pairs(str(facts_path), split, pairs_per_episode)
            traces = canonical_traces(make_continuous_traces(pairs, 3))
        write_jsonl(str(trace_path), traces)
        paths[split] = trace_path.name
        audits[split] = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'source_rows': sum(1 for line in path.read_text(encoding='utf-8').splitlines() if line.strip()),
                         'owners_before_token_filter': len(traces), 'model_inputs_sha256': trace_digest(traces)}
        episodes_path.unlink()
        facts_path.unlink()
    manifest = {'trace_paths': paths, 'source': 'MSC', 'pairs_per_episode': pairs_per_episode,
                'preprocessing': {'profile_character_cap': 220, 'dialogue_character_cap': 260,
                                  'whitespace_normalized': True},
                'paper_cohort_verified': paper_cohort,
                'fact_limits': {'train': 0 if paper_cohort else train_fact_limit,
                                'validation': 0 if paper_cohort else validation_fact_limit},
                'audit': audits}
    target = output / 'manifest.json'
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return target




def reconstruct_cohort(facts, cohort):
    """Recover published model inputs from source IDs, failing on any mismatch."""
    lookup = {row['record_id']: row for row in facts}
    if len(lookup) != len(facts):
        raise ValueError('Duplicate source record IDs')
    traces = []
    for row in cohort['traces']:
        steps = []
        for step in row['steps']:
            events = []
            for member in step['events']:
                record_id = member['record_id']
                if record_id not in lookup:
                    raise ValueError('Missing paper source record ID: ' + record_id)
                fact = lookup[record_id]
                event = atomic_record(fact, 'G' if fact['task'] == 'global' else 'D')
                event = {key: event[key] for key in ('owner_id', 'text', 'label', 'value')}
                digest = hashlib.sha256(json.dumps(event, ensure_ascii=False, sort_keys=True,
                                                   separators=(',', ':')).encode('utf-8')).hexdigest()
                if digest != member['sha256']:
                    raise ValueError('Paper record hash mismatch: ' + record_id)
                events.append(event)
            steps.append({'step': step['step'], 'events': events})
        traces.append({'owner_id': row['owner_id'], 'steps': steps})
    if trace_digest(traces) != cohort['model_inputs_sha256']:
        raise ValueError('Paper model-input hash mismatch')
    return traces

def canonical_traces(rows):
    """Retain exactly the ordered fields consumed by the HasMem loader."""
    return [{"owner_id": str(row["owner_id"]),
             "steps": [{"step": int(step.get("step", 0)),
                        "events": [{key: event[key] for key in ("owner_id", "text", "label", "value")}
                                   for event in step.get("events", [])]}
                       for step in row.get("steps", [])]} for row in rows]


def trace_digest(rows):
    text = json.dumps(canonical_traces(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_traces(source_dir, output_dir, verify_paper=False):
    """Import the preserved paper traces without altering model inputs."""
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    expected = {
        "train": "2e15452287a005161fc68a15157a53b193c5da1a616a95fcad109feb0f89f52c",
        "validation": "7c9050c1e9eca1538eb28f519c0a39dc7b3382d52bd56068b4d075ea2d87f469",
    }
    prepared, audits = {}, {}
    for split in ("train", "validation"):
        path = source / f"GLOBAL_PROMOTION_TRACE_{split.upper()}.jsonl"
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if verify_paper and checksum != expected[split]:
            raise ValueError(f"{split} trace hash differs from the paper source")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        prepared[split] = canonical_traces(rows)
        audits[split] = {"source_sha256": checksum, "model_inputs_sha256": trace_digest(rows),
                         "owners_before_token_filter": len(rows),
                         "events_before_token_filter": sum(len(step["events"]) for row in rows for step in row["steps"])}
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for split, rows in prepared.items():
        name = f"{split}.traces.jsonl"
        write_jsonl(str(output / name), rows)
        paths[split] = name
    manifest = {"trace_paths": paths, "source": "MSC", "paper_source_verified": verify_paper, "audit": audits}
    target = output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-dir', help='Raw split-specific MSC JSONL directory')
    source.add_argument('--trace-dir', help='Preserved GLOBAL_PROMOTION_TRACE_*.jsonl directory')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pairs-per-episode', type=int, default=2)
    parser.add_argument('--new-cohort', action='store_true', help='Build a new cohort instead of the verified paper membership')
    parser.add_argument('--train-fact-limit', type=int, default=0)
    parser.add_argument('--validation-fact-limit', type=int, default=0)
    parser.add_argument('--verify-paper-traces', action='store_true')
    args = parser.parse_args()
    if args.trace_dir:
        print(import_traces(args.trace_dir, args.output_dir, args.verify_paper_traces))
    else:
        if args.verify_paper_traces:
            parser.error('--verify-paper-traces requires --trace-dir')
        print(prepare(args.source_dir, args.output_dir, args.pairs_per_episode,
                      args.train_fact_limit, args.validation_fact_limit, paper_cohort=not args.new_cohort))


if __name__ == '__main__':
    main()
