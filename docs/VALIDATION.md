# Software validation

The local release was checked with Python 3.13, PyTorch 2.7.1+cu118, Transformers 5.5.4, and an NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB). Full pretrained-model experiments were not rerun as part of these software checks.

```bash
python -m compileall -q hasmem tests
python -m hasmem check-config --config configs/fixed_steps.json
python -m unittest discover -s tests -v
```

The suite includes 72 checks across causal rollout, KEEP identity, width limits, SHRINK/EXPAND behavior, constrained search, native assistant termination, deterministic MSC preparation, and configuration/checkpoint validation, local judge coverage, and fail-closed cohort reconstruction. The CUDA integration check creates a random one-layer Qwen2 model and a constructed dataset; it performs warmup and policy updates, saves and restores the checkpoint, evaluates both hard and adaptive conditions, and runs the public local-record inference CLI. It skips on hosts without CUDA bfloat16 support.

The portable entry point removes cluster-specific scheduler/path requirements. Data and model paths are supplied by configuration. Explicit `return_dict=False` keeps chat-template token sequences consistent across supported Transformers versions. Checkpoints load with PyTorch's restricted `weights_only=True` mode. The training equations, native chat framing, greedy decoding, and core action/width rules are retained.

The paper MSC train and development traces were matched to their experiment-audit SHA-256 hashes. All 12,698 training and 662 development records were reconstructed from preserved source-event fields and verified against the text-free cohort manifests. The imported development cases exactly match the original cases under Qwen2.5-7B tokenization (268 owners, 646 records, 535 questions). The complete raw-MSC-export-to-trace path was not rerun because that source export was unavailable. These checks do not evaluate a pretrained language model.
