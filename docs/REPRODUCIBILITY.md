# Reproduction protocols

## Main configuration versus fixed-step experiments

The main HasMem result uses a wall-clock curriculum and the final checkpoint of seed 2026091331. That run completed 5,465 total updates. Its MSC-derived F1/EM/NLL are 95.3/84.3/0.115, with a relative memory-position count of 0.936. These are published reference values, not scores produced by installing this repository.

The cross-seed experiment uses a separate fixed-step curriculum: 384 warmup updates plus 5,081 policy updates, with checkpoint selection by training loss inside the non-KEEP EMA band [0.22, 0.40]. For seed 2026091331, the selected checkpoint is update 385 and remains all KEEP. The two experiments share a seed but use different schedules and checkpoints.

| Paper experiment | Recipe | Additional setting |
| --- | --- | --- |
| Main HasMem/hard comparison | `paper_main_wallclock.json` | Evaluate final checkpoint on all eligible development owners and all 500 LongMemEval-S questions |
| Fixed-step full model | `fixed_steps.json` | Verify 5,465 completed updates |
| Cross-seed selection study | `multiseed_selection.json` | Seeds 2026091331, 2026091401, 2026091701, 2026091702, 2026091703, 2026091711, 2026091712, 2026091713 |
| Target-band strength scan | `fixed_steps.json` | Set `method.band_weight` to 0, 0.6, 1.5, 3, or 6; choose the corresponding backbone |
| Training-component ablation | `fixed_steps.json` as a starting recipe | Explicitly set `force`, `band`, `hinge`, `writer`, `use_reader`, or `use_global`; retain the intended training budget and selection rule |

Published ablation runs include different completed update counts and training budgets. Re-running all variants with the fixed-step recipe defines a new controlled comparison; it does not reproduce those individual historical run identities. The paper tables specify each reported setting.

## Model and data identity

The original trained adapter checkpoints are not bundled. Exact reproduction of a saved paper output requires the original checkpoint and tokenizer/data revisions. Local reruns execute the released algorithm, but hardware, framework kernels, time-based scheduling, and checkpoint selection can change the result. Record model revisions, package versions, seeds, input hashes, and completed update counts alongside each run.

Main default settings are a 64-dimensional memory state and encoder feature, rank-4 residuals on the final four attention output projections, a minimum slot width of 4, a 10% width adjustment, learning rate 1e-4, and 384 warmup updates. The configured non-KEEP target decreases from 0.50 to 0.22; forced-softening probability decreases from 0.75 to 0.15. Global and Reader parameters are frozen after warmup, while the Controller and enabled Writer continue training.

The fixed-step trainer uses eight owners per optimizer update. Larger models use gradient accumulation. The wall-clock trainer preserves its original batching rule; for the 7B main model with retrieval width 8, this is two physical owners and four accumulation steps.

## Metrics

MSC-derived paper metrics average questions within owner and then average owners. LongMemEval-S table metrics average its 500 questions. Both forms are explicitly available in `SUMMARY.json`; per-question records remain in `rows.jsonl`.

EM compares normalized answer strings. Cover tests answer coverage with the question-type rules in `hasmem/metrics.py`. Brief additionally requires normalized output length at most `max(12, 3 * gold_length)`. These local diagnostics and answer NLL measure different properties. Run `python -m hasmem.judge` for the local Qwen-judge procedure. It preserves the historical yes/no parser and task-specific official prompt text; the reference LongMemEval evaluator uses a different judge configuration.

Position budget counts sequence positions, including role framing where specified; it is not a direct measurement of byte storage, peak GPU memory, or latency. The output records those quantities separately when available.

## Validation scope

The release includes unit tests of actual rollout and action-search code plus a CUDA integration test with a randomly initialized tiny Qwen2 model. These checks validate software paths and invariants. They do not reproduce the paper's pretrained-model benchmark scores.
