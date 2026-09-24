"""Group-paired, seed-conditional summaries; no I/O and no raw identifiers returned."""

from collections import defaultdict
import math
import random


METRICS = ("nll", "f1", "em", "memory_vectors", "cumulative_vectors")
REFERENCES = ("hard", "fixed", "random", "noshrink")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260913
REQUIRED = {
    "dataset", "model", "group", "case_id", "condition", "old", "seed", *METRICS
}


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def _validate(rows):
    checked = []
    seen = set()
    for row in rows:
        missing = REQUIRED.difference(row)
        if missing:
            raise ValueError("Missing required fields: " + ", ".join(sorted(missing)))
        if not isinstance(row["old"], bool):
            raise ValueError("old must be a boolean")
        for label in ("dataset", "model", "condition"):
            if not isinstance(row[label], str) or not row[label]:
                raise ValueError(label + " must be a nonempty string")
        for metric in METRICS:
            value = row[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(metric + " must be numeric")
            if not math.isfinite(value):
                raise ValueError(metric + " must be finite")
        key = tuple(row[k] for k in (
            "model", "dataset", "condition", "group", "case_id", "seed"
        ))
        try:
            if key in seen:
                raise ValueError("Duplicate model/dataset/condition/group/case/seed row")
            seen.add(key)
        except TypeError as exc:
            raise ValueError("group, case_id, and seed must be hashable scalars") from exc
        checked.append(dict(row))
    return checked


def _group_vectors(rows):
    """Equal cases within each group/seed, then equal observed seeds per group."""
    by_group_seed = defaultdict(list)
    for row in rows:
        by_group_seed[(row["group"], row["seed"])].append(row)
    by_group = defaultdict(list)
    for (group, _seed), cases in by_group_seed.items():
        by_group[group].append([_mean(r[m] for r in cases) for m in METRICS])
    return {
        group: [_mean(v[i] for v in vectors) for i in range(len(METRICS))]
        for group, vectors in by_group.items()
    }


def _point_summary(rows):
    vectors = list(_group_vectors(rows).values())
    return {
        "n_rows": len(rows),
        "n_groups": len(vectors),
        "n_cases": len({(r["group"], r["case_id"]) for r in rows}),
        "n_seeds": len({r["seed"] for r in rows}),
        "mean": {
            metric: _mean(v[i] for v in vectors)
            for i, metric in enumerate(METRICS)
        },
    }


def _percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _paired_effect(rows):
    grouped = _group_vectors(rows)
    # Stable ordering makes output independent of input row ordering.
    vectors = [grouped[k] for k in sorted(grouped, key=lambda x: (type(x).__name__, str(x)))]
    count = len(vectors)
    if not count:
        return {"n_groups": 0, "n_rows": 0, "effects": None,
                "inference_warning": "no_matched_observations"}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [[] for _ in METRICS]
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [vectors[rng.randrange(count)] for _ in range(count)]
        for i in range(len(METRICS)):
            samples[i].append(_mean(v[i] for v in selected))
    return {
        "n_groups": count,
        "n_rows": len(rows),
        "effects": {
            metric: {
                "delta": _mean(v[i] for v in vectors),
                "ci95": [_percentile(samples[i], .025), _percentile(samples[i], .975)],
            }
            for i, metric in enumerate(METRICS)
        },
        "inference_warning": "only_one_independent_group" if count < 2 else None,
    }


def _pair(adaptive, reference):
    def index(rows):
        return {(r["group"], r["case_id"], r["seed"]): r for r in rows}

    left, right = index(adaptive), index(reference)
    common = left.keys() & right.keys()
    union = left.keys() | right.keys()
    missing_left, missing_right = right.keys() - left.keys(), left.keys() - right.keys()
    complete = bool(common) and not missing_left and not missing_right
    differences = []
    for key in sorted(common, key=repr):
        if left[key]["old"] != right[key]["old"]:
            raise ValueError("Paired observations disagree on old/new classification")
        row = {k: left[key][k] for k in ("group", "case_id", "seed", "old")}
        row.update({m: left[key][m] - right[key][m] for m in METRICS})
        differences.append(row)
    observed_groups = {key[0] for key in union}
    matched_groups = {key[0] for key in common}
    result = {
        "status": "complete" if complete else "missing_coverage",
        "coverage_complete": complete,
        "coverage": {
            "adaptive_rows": len(left), "reference_rows": len(right),
            "union_rows": len(union), "matched_rows": len(common),
            "missing_adaptive_rows": len(missing_left),
            "missing_reference_rows": len(missing_right),
            "union_groups": len(observed_groups), "matched_groups": len(matched_groups),
            "groups_with_unmatched_rows": len({k[0] for k in missing_left | missing_right}),
        },
        "estimand": "adaptive_minus_reference_on_exactly_matched_case_and_seed_rows",
        "n_matched_seeds": len({k[2] for k in common}),
        "all": _paired_effect(differences),
        "old": _paired_effect([r for r in differences if r["old"]]),
        "new": _paired_effect([r for r in differences if not r["old"]]),
    }
    result["full_coverage_inference_available"] = complete and len(matched_groups) >= 2
    return result


def summarize(rows):
    """Return JSON-compatible aggregate results without group/case IDs or raw rows.

    Pairing requires identical model, dataset, group, case_id, and seed. Missing
    coverage is explicit; any partial paired CI estimates only the intersection.
    The function cannot infer cases missing from *every* supplied condition; callers
    must additionally compare these counts with their frozen evaluation manifest.
    """
    rows = _validate(rows)
    partitions = defaultdict(list)
    for row in rows:
        partitions[(row["model"], row["dataset"], row["condition"])].append(row)
    aggregates = []
    for (model, dataset, condition), partition in sorted(partitions.items()):
        aggregates.append({
            "model": model, "dataset": dataset, "condition": condition,
            "all": _point_summary(partition),
            "old": _point_summary([r for r in partition if r["old"]]),
            "new": _point_summary([r for r in partition if not r["old"]]),
        })
    comparisons = []
    for model, dataset in sorted({(r["model"], r["dataset"]) for r in rows}):
        adaptive = partitions.get((model, dataset, "adaptive"), [])
        for reference in REFERENCES:
            if (model, dataset, reference) not in partitions:
                continue
            pair = _pair(adaptive, partitions.get((model, dataset, reference), []))
            pair.update({"model": model, "dataset": dataset,
                         "condition": "adaptive", "reference": reference})
            comparisons.append(pair)
    return {
        "schema_version": 1,
        "status": "empty" if not rows else (
            "complete" if all(p["coverage_complete"] for p in comparisons) else "missing_coverage"
        ),
        "n_rows": len(rows),
        "aggregation": "equal cases within group/seed; equal observed seeds within group; equal groups",
        "bootstrap": {"unit": "paired_group", "replicates": BOOTSTRAP_REPLICATES,
                      "seed": BOOTSTRAP_SEED, "interval": "percentile_95_unadjusted"},
        "uncertainty_scope": "conditional on observed trained models/seeds; not seed-population or retraining uncertainty",
        "direction": {"nll": "lower_is_better", "f1": "higher_is_better", "em": "higher_is_better",
                      "memory_vectors": "lower_cost", "cumulative_vectors": "lower_cost"},
        "manifest_warning": "coverage checks only supplied rows; verify expected cases/seeds against the frozen manifest",
        "multiplicity_warning": "CIs are descriptive and unadjusted across metrics/comparisons",
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
    }
