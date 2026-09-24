# Software validation

The local release was checked with Python 3.13, PyTorch 2.7.1+cu118, Transformers 5.5.4, and an NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB). Full pretrained-model experiments were not rerun as part of these software checks.

```bash
python -m compileall -q hasmem tests
# After preparing data and setting the configuration paths:
python -m hasmem check-config --config configs/fixed_steps.json
python -m unittest discover -s tests -v
```

The suite includes 86 checks across causal rollout, KEEP identity, width limits, SHRINK/EXPAND behavior, constrained search, native assistant termination, deterministic MSC preparation, and configuration/checkpoint validation, local judge coverage, fail-closed cohort reconstruction, input preflight, optional model-revision pinning, and verified dataset download. The CUDA integration check creates a random one-layer Qwen2 model and a constructed dataset; it performs warmup and policy updates, saves and restores the checkpoint, evaluates both hard and adaptive conditions, and runs the public local-record inference CLI. It skips on hosts without CUDA bfloat16 support.

The portable entry point removes cluster-specific scheduler/path requirements. Data and model paths are supplied by configuration. Explicit `return_dict=False` keeps chat-template token sequences consistent across supported Transformers versions. Checkpoints load with PyTorch's restricted `weights_only=True` mode. The training equations, native chat framing, greedy decoding, and core action/width rules are retained.

The pinned MSC mirror was downloaded and prepared through the public CLI. All 12,698 training and 662 development records, including their complete ordered model inputs, match the preserved paper traces and cohort manifests. The prepared development cases yield 268 owners, 646 records, and 535 questions under the paper's Qwen2.5-7B token filter. Source revisions, downloaded file hashes, and canonical input hashes are recorded in [msc_provenance.json](msc_provenance.json). These data checks do not evaluate a pretrained language model.
