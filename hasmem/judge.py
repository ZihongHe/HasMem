"""Local LongMemEval judging with the paper's Qwen yes/no protocol.

Prompt text is from LongMemEval (MIT, Copyright 2024 Di Wu).
Source: https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
See third_party/LongMemEval-LICENSE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a subset "
                "of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
                "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a subset "
                "of the information required by the answer, answer no. In addition, do not penalize off-by-one "
                "errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
                "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the "
                "model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
                "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the response "
                "should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a "
                "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as long "
                "as it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(task)
    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. The model could say "
        "that the information is incomplete, or some other information is given but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)

def parse_yes(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith(("no", "\u5426", "incorrect", "wrong")):
        return False
    if lowered.startswith(("yes", "\u662f", "correct")):
        return True
    return "yes" in lowered


def load_inputs(rows_path, refs_path, allow_subset=False):
    refs_list = json.loads(Path(refs_path).read_text(encoding="utf-8"))
    refs = {str(row["question_id"]): row for row in refs_list}
    if len(refs) != len(refs_list):
        raise ValueError("Reference question IDs must be unique")
    if not allow_subset and len(refs) != 500:
        raise ValueError("Full LongMemEval-S judging requires 500 reference questions")
    groups = {}
    for line in Path(rows_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") != "longmemeval_s":
            continue
        case_id = str(row["case_id"])
        if not case_id.startswith("lme_s:") or not case_id.endswith("/0"):
            raise ValueError("Unexpected LongMemEval case ID: " + case_id)
        qid = case_id[len("lme_s:"):-2]
        condition = str(row["condition"])
        if not condition or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in condition):
            raise ValueError("Unsafe condition name")
        group = groups.setdefault(condition, {})
        if qid in group:
            raise ValueError("Duplicate prediction for " + condition + ": " + qid)
        if qid not in refs:
            raise ValueError("Prediction has no reference: " + qid)
        group[qid] = str(row.get("prediction") or "")
    if not groups:
        raise ValueError("No LongMemEval-S prediction rows")
    expected = None
    for condition, predictions in groups.items():
        ids = set(predictions)
        if not allow_subset and ids != set(refs):
            raise ValueError("Incomplete prediction coverage for " + condition)
        if expected is not None and ids != expected:
            raise ValueError("Conditions must cover identical question IDs")
        expected = ids
    return refs, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="HasMem evaluation rows.jsonl")
    parser.add_argument("--refs", required=True, help="longmemeval_s_cleaned.json")
    parser.add_argument("--model", required=True, help="Qwen2.5-7B-Instruct model ID or directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--allow-subset", action="store_true", help="Explicit diagnostic subset; each condition must have identical IDs")
    args = parser.parse_args()
    refs, groups = load_inputs(args.rows, args.refs, args.allow_subset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Local Qwen judging requires a CUDA GPU")
    options = dict(revision=args.revision, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **options).cuda().eval()
    metadata = {"judge_model": args.model, "revision": args.revision,
                "protocol": "official_longmemeval_prompts_local_qwen",
                "decoding": {"do_sample": False, "max_new_tokens": 10},
                "diagnostic_subset": args.allow_subset,
                "prompt_source": "https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py",
                "rows_sha256": hashlib.sha256(Path(args.rows).read_bytes()).hexdigest(),
                "refs_sha256": hashlib.sha256(Path(args.refs).read_bytes()).hexdigest()}
    (output / "CONFIG.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    aggregate = {}
    started = time.time()
    with torch.inference_mode():
        for condition, predictions in groups.items():
            labels, by_type = [], {}
            with (output / (condition + ".jsonl")).open("w", encoding="utf-8") as handle:
                for index, (qid, hypothesis) in enumerate(predictions.items(), 1):
                    ref = refs[qid]
                    prompt = get_anscheck_prompt(ref["question_type"], ref["question"], ref["answer"],
                                                 hypothesis, abstention="_abs" in qid)
                    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                        tokenize=True, return_dict=False, add_generation_prompt=True)
                    inp = torch.tensor([ids], device="cuda")
                    generated = model.generate(inp, max_new_tokens=10, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    text = tokenizer.decode(generated[0, inp.shape[1]:], skip_special_tokens=True).strip()
                    label = parse_yes(text)
                    labels.append(int(label))
                    by_type.setdefault(ref["question_type"], []).append(int(label))
                    row = {"question_id": qid, "question_type": ref["question_type"], "hypothesis": hypothesis,
                           "autoeval_label": {"model": args.model, "label": label, "raw": text}}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if index % 25 == 0 or index == len(predictions):
                        print(f"{condition}: {index}/{len(predictions)}", flush=True)
            summary = {"n": len(labels), "accuracy": sum(labels) / len(labels),
                       "by_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())},
                       "type_n": {k: len(v) for k, v in sorted(by_type.items())},
                       "judge_model": args.model, "protocol": metadata["protocol"]}
            (output / (condition + ".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            aggregate[condition] = summary
    (output / "AGGREGATE.json").write_text(json.dumps({"seconds": time.time() - started, "cells": aggregate}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
