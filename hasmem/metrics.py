"""Local lexical, coverage, and brevity diagnostics for LongMemEval-S."""
from __future__ import annotations

import re
import string
from collections import Counter

ARTICLES = re.compile(r"\b(a|an|the)\b")
SPACE = re.compile(r"\s+")
PUNCT = str.maketrans("", "", string.punctuation)
ABS_CUES = (
    "unknown", "unanswerable", "not mentioned", "not enough", "cannot answer",
    "no information", "incomplete", "not in the memory", "i don't know",
    "do not know", "doesn't mention", "does not mention",
)


def norm(text):
    return SPACE.sub(" ", ARTICLES.sub(" ", str(text).lower().translate(PUNCT))).strip()


def tokens(text):
    return norm(text).split()


def first_span(pred):
    text = str(pred or "").strip()
    if not text:
        return ""
    return re.split(r"(?<=[.!?\u3002\uff01\uff1f\n])\s+", text, maxsplit=1)[0].strip()


def gold_text(gold):
    if isinstance(gold, list):
        return "; ".join(str(x) for x in gold)
    return "" if gold is None else str(gold)


def gold_parts(gold):
    if isinstance(gold, list):
        return [str(x).strip() for x in gold if str(x).strip()]
    text = str(gold)
    if "; " in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def token_prf(pred, gold):
    pred_toks, gold_toks = tokens(pred), tokens(gold_text(gold))
    if not pred_toks and not gold_toks:
        return 1.0, 1.0, 1.0
    if not pred_toks or not gold_toks:
        return 0.0, 0.0, 0.0
    hit = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    return hit / len(pred_toks), hit / len(gold_toks), 2 * hit / (len(pred_toks) + len(gold_toks))


def _part_covered(pred_norm, pred_toks, part, question_type):
    gold_n = norm(part)
    if not gold_n:
        return False
    if gold_n in pred_norm:
        return True
    gold_toks = gold_n.split()
    if gold_toks and all(tok in pred_toks for tok in gold_toks):
        return True
    if question_type == "temporal-reasoning":
        nums = re.findall(r"-?\d+", gold_n)
        if len(nums) == 1:
            target = int(nums[0])
            pred_nums = [int(x) for x in re.findall(r"-?\d+", pred_norm)]
            others = [tok for tok in gold_toks if not re.fullmatch(r"-?\d+", tok)]
            if any(abs(value - target) <= 1 for value in pred_nums) and all(tok in pred_toks for tok in others):
                return True
    return False


def answer_covered(pred, gold, question_type, question_id=""):
    """Binary: full gold appears in the prediction. Extra text allowed."""
    if "_abs" in str(question_id):
        lowered = norm(pred)
        return any(cue in lowered for cue in ABS_CUES)
    pred_n = norm(pred)
    pred_toks = set(pred_n.split())
    if question_type == "single-session-preference":
        gold_toks = [tok for tok in tokens(gold_text(gold)) if len(tok) > 2]
        if not gold_toks:
            return _part_covered(pred_n, pred_toks, gold_text(gold), question_type)
        return sum(tok in pred_toks for tok in gold_toks) >= max(1, (len(gold_toks) + 1) // 2)
    return all(_part_covered(pred_n, pred_toks, part, question_type) for part in gold_parts(gold))


def brief_prediction(pred, gold):
    gold_n = max(1, len(tokens(gold_text(gold))))
    return len(tokens(pred)) <= max(12, 3 * gold_n)


def score_prediction(pred, gold, question_type="", question_id=""):
    precision, recall, f1 = token_prf(pred, gold)
    cover = answer_covered(pred, gold, question_type, question_id)
    span = answer_covered(first_span(pred), gold, question_type, question_id)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "em": float(norm(pred) == norm(gold_text(gold))),
        "cover": float(cover),
        "span_cover": float(span),
        "brief_cover": float(cover and brief_prediction(pred, gold)),
    }
