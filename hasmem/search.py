"""Small, deterministic fit-only counterfactual beam search.

This is bounded beam search, not evolutionary optimization or ant-colony
optimization. Each partial plan is evaluated as a *complete* plan padded with
KEEP. All evaluations must use the same causal memory prefix and questions.
The caller owns model execution and must not use held-out answers here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


KEEP = 0
SHRINK10 = 1
EXPAND10 = 2
ACTIONS = (KEEP, SHRINK10, EXPAND10)


def _finite_number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def beam_search(
    depth: int,
    score_fn: Callable[[tuple[int, ...]], Mapping[str, Any]],
    quality_tolerance: float = 1e-4,
    cost_weight: float = 0.05,
    beam_width: int = 2,
    *,
    hard_teacher_nll: Sequence[float] | None = None,
    harm_weight: float = 4.0,
    keep_penalty_cap: float = 0.02,
) -> dict[str, Any]:
    """Return one safe beneficial compression target, or no positive target.

    ``score_fn`` receives a full plan of length ``depth`` and returns:

    * ``per_question_nll``: comparable question losses in a fixed order;
    * ``cumulative_cost``: cumulative cost, already normalized by the caller
      against one fixed hard-history reference for every candidate;
    * ``keep_streak_penalty``: a nonnegative proposed penalty, capped here.
    * optional ``realized_sequence``: actual actions after capacity constraints
      convert unavailable shrink/expand requests to KEEP. Defaults to the
      requested plan for backwards compatibility.

    The quality-cost score is ``mean(nll + harm_weight * relu(nll-teacher))
    + cost_weight * cumulative_cost``. If no teacher vector is supplied, the
    all-KEEP plan provides it. The capped KEEP penalty is added for ranking,
    but cannot override the separate safety eligibility conditions.

    The action space is KEEP (0), SHRINK10 (1), and EXPAND10 (2). Expansion
    only resizes the retained memory state; it must not restore archived text.
    A positive target must (i) contain an explicit SHRINK10 action, (ii) have no question
    NLL worse than the all-KEEP plan by more than ``quality_tolerance``,
    (iii) save strictly positive cumulative cost, and (iv) improve the total
    objective. This is a fit-only numerical safeguard, not a guarantee on
    generation quality or on held-out/future questions.

    Ties are broken by the full action tuple. At depth two and beam width
    two, at most seven unique plans are evaluated. This is not exhaustive
    search over all nine depth-two plans. The selected positive is the best
    eligible plan among *all evaluated* full plans, including KEEP-padded
    plans seen before the final beam expansion.

    ``action_target`` is the first realized non-KEEP action in the winning plan;
    ``first_divergence`` is its zero-based decision index. Train against the
    corresponding causal decision state, not blindly against the first
    state. The complete requested ``winner_sequence`` is returned for exact
    replay, alongside ``winner_realized_sequence`` for action auditing.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 2:
        raise ValueError("depth must be an integer in [1, 2]")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or not 1 <= beam_width <= 2:
        raise ValueError("beam_width must be an integer in [1, 2]")
    quality_tolerance = _finite_number(quality_tolerance, "quality_tolerance")
    cost_weight = _finite_number(cost_weight, "cost_weight")
    harm_weight = _finite_number(harm_weight, "harm_weight")
    keep_penalty_cap = _finite_number(keep_penalty_cap, "keep_penalty_cap")
    if min(quality_tolerance, cost_weight, harm_weight, keep_penalty_cap) < 0:
        raise ValueError("tolerance, weights, and penalty cap must be nonnegative")

    reference_sequence = (KEEP,) * depth
    raw_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def evaluate_raw(sequence: tuple[int, ...], stage: int) -> dict[str, Any]:
        if sequence in raw_cache:
            return raw_cache[sequence]
        raw = score_fn(sequence)
        nll = [_finite_number(x, "per_question_nll") for x in raw["per_question_nll"]]
        if not nll:
            raise ValueError("per_question_nll must contain at least one question")
        if any(x < 0 for x in nll):
            raise ValueError("per_question_nll must be nonnegative")
        cost = _finite_number(raw["cumulative_cost"], "cumulative_cost")
        penalty = _finite_number(raw.get("keep_streak_penalty", 0.0), "keep_streak_penalty")
        if cost < 0 or penalty < 0:
            raise ValueError("cumulative_cost and keep_streak_penalty must be nonnegative")
        realized = list(raw.get("realized_sequence", sequence))
        if len(realized) != depth or any(
            isinstance(action, bool) or not isinstance(action, int) or action not in ACTIONS
            for action in realized
        ):
            raise ValueError("realized_sequence must match the plan length and action space")
        record = {
            "sequence": list(sequence),
            "realized_sequence": realized,
            "evaluated_at_depth": stage,
            "per_question_nll": nll,
            "cumulative_cost": cost,
            "raw_keep_streak_penalty": penalty,
            "keep_streak_penalty": min(penalty, keep_penalty_cap),
        }
        raw_cache[sequence] = record
        return record

    reference = evaluate_raw(reference_sequence, 0)
    teacher = list(reference["per_question_nll"]) if hard_teacher_nll is None else [
        _finite_number(x, "hard_teacher_nll") for x in hard_teacher_nll
    ]
    if len(teacher) != len(reference["per_question_nll"]):
        raise ValueError("hard_teacher_nll must match the reference question count")
    if any(x < 0 for x in teacher):
        raise ValueError("hard_teacher_nll must be nonnegative")

    def evaluate(sequence: tuple[int, ...], stage: int) -> dict[str, Any]:
        record = evaluate_raw(sequence, stage)
        if "objective" not in record:
            nll = record["per_question_nll"]
            if len(nll) != len(teacher):
                raise ValueError("Every plan must score the same number of questions")
            quality = math.fsum(
                current + harm_weight * max(0.0, current - target)
                for current, target in zip(nll, teacher)
            ) / len(nll)
            base_objective = quality + cost_weight * record["cumulative_cost"]
            record.update(
                quality_objective=quality,
                base_objective=base_objective,
                objective=base_objective + record["keep_streak_penalty"],
            )
        return record

    reference = evaluate(reference_sequence, 0)
    beam: list[tuple[int, ...]] = [()]
    beam_history = []
    for level in range(1, depth + 1):
        candidates = []
        for prefix in beam:
            for action in ACTIONS:
                candidate_prefix = prefix + (action,)
                full_plan = candidate_prefix + (KEEP,) * (depth - level)
                record = evaluate(full_plan, level)
                candidates.append((record["objective"], full_plan, candidate_prefix))
        candidates.sort(key=lambda item: (item[0], item[1]))
        beam = [item[2] for item in candidates[:beam_width]]
        beam_history.append({"depth": level, "prefixes": [list(prefix) for prefix in beam]})

    qualified: list[tuple[float, tuple[int, ...]]] = []
    nonkeep: list[tuple[float, tuple[int, ...]]] = []
    for sequence, record in raw_cache.items():
        # All cached plans have been fully scored by evaluate().
        question_delta = [
            current - original
            for current, original in zip(record["per_question_nll"], reference["per_question_nll"])
        ]
        cost_savings = reference["cumulative_cost"] - record["cumulative_cost"]
        advantage = reference["objective"] - record["objective"]
        base_advantage = reference["base_objective"] - record["base_objective"]
        has_compression = any(action == SHRINK10 for action in record["realized_sequence"])
        quality_safe = all(delta <= quality_tolerance for delta in question_delta)
        eligible = has_compression and quality_safe and cost_savings > 0 and advantage > 0
        record.update(
            per_question_delta_vs_keep=question_delta,
            max_question_harm=max(question_delta),
            cost_savings_vs_keep=cost_savings,
            advantage_vs_keep=advantage,
            base_advantage_vs_keep=base_advantage,
            quality_safe=quality_safe,
            positive_eligible=eligible,
        )
        if any(action != KEEP for action in sequence):
            nonkeep.append((record["objective"], sequence))
        if eligible:
            qualified.append((record["objective"], sequence))

    qualified.sort(key=lambda item: (item[0], item[1]))
    nonkeep.sort(key=lambda item: (item[0], item[1]))
    winner = qualified[0][1] if qualified else reference_sequence
    winning_record = raw_cache[winner]
    realized_winner = winning_record["realized_sequence"]
    divergence = next((index for index, action in enumerate(realized_winner) if action != KEEP), None) if qualified else None
    return {
        "method": "deterministic_keep_padded_beam_search",
        "depth": depth,
        "beam_width": beam_width,
        "evaluated_count": len(raw_cache),
        "reference_sequence": list(reference_sequence),
        "reference_objective": reference["objective"],
        "winner_sequence": list(winner),
        "winner_realized_sequence": realized_winner,
        "positive": bool(qualified),
        "advantage": winning_record["advantage_vs_keep"],
        "base_advantage": winning_record["base_advantage_vs_keep"],
        "first_divergence": divergence,
        "action_target": realized_winner[divergence] if divergence is not None else None,
        "best_nonkeep_sequence": list(nonkeep[0][1]) if nonkeep else None,
        "qualified_count": len(qualified),
        "quality_tolerance": quality_tolerance,
        "cost_weight": cost_weight,
        "harm_weight": harm_weight,
        "keep_penalty_cap": keep_penalty_cap,
        "beam_history": beam_history,
        "records": list(raw_cache.values()),
        "reason": "qualified_compression" if qualified else "no_safe_beneficial_compression_evaluated",
    }
