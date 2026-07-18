# MiniCPM-o 4.5 Ascend core companion branch

This branch pins SGLang core to the exact source lineage used by the HiDevLab
Ascend image. It is the model-loader/backend companion to the SGLang-Omni
competition branch, not a rolling mirror of SGLang main.

## Locked baseline

- Branch: `codex/minicpmo45-ascend-competition-core`
- Base: `f308abc05212c2f5f455de22a525e14afa63ee4f`
- Image package: `sglang 0.5.14.dev172+gf308abc05`
- Runtime: CANN 9.0.0, torch/torch_npu 2.10.0, transformers 5.8.1

The image working tree replaces `python/pyproject.toml` with the committed NPU
dependency variant and deletes `python/pyproject_npu.toml`. Preserve that state
as a vendor patch/manifest; do not mix it into MiniCPM model commits.

## Current model boundary

At the base commit, `python/sglang/srt/models/minicpmo.py` is the MiniCPM-o 2.6
Qwen2 path and sets `init_tts = False`. It is not MiniCPM-o 4.5 support.

## Planned commit layers

1. Ascend dependency/install contract and image-overlay fixture.
2. Device-safe MiniCPM resampler/encoder code; no hard-coded CUDA.
3. Qwen3/version dispatch and text/image/audio input→text correctness.
4. Hidden-state export contract for the Omni stage bridge.
5. MiniCPMTTS hidden→S3 support and fixtures.
6. NPU correctness/performance tests with exact runtime manifests.

Keep this branch pinned and cherry-pick deliberately. If a new official image
changes the SGLang SHA, torch_npu or CANN contract, audit it before rebasing or
create a versioned successor branch.
