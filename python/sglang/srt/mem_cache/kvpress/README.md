# KVPress for SGLang — KV-cache compression aligned with the NVIDIA reference

Per-request **KV-cache compression** for SGLang. After prefill, each request's KV is pruned to a
fraction of its tokens, freeing pool slots so long contexts cost less memory. The scoring methods
are ported from and validated against **[NVIDIA kvpress](https://github.com/NVIDIA/kvpress)**.

Five **attention-free** presses are supported: `knorm`, `random`, `streamingllm`, `keydiff`, `lagkv`.
(Attention-weight presses like SnapKV/ExpectedAttention are not — see *Limitations*.)

---

## TL;DR

- **What it does:** drops the least-important `ratio` fraction of prefill KV tokens, per request,
  right after prefill; decode then continues on the compacted cache.
- **Alignment:** behaviorally matched to NVIDIA kvpress-on-HF — end-to-end greedy-generation
  ROUGE-L between *SGLang-compressed* and *NV-compressed* outputs is **≈0.99** on long context for all
  four deterministic presses; the running engine's real per-layer kept-token set matches NV's
  per-head selection at the same Jaccard the offline analysis predicts; `ratio=0` reproduces the
  uncompressed baseline token-for-token.
- **Hard limitation:** SGLang's paged pool is **per-token, not per-head** (one slot = one token across
  all heads), so it keeps a single shared token set where NV prunes each head independently. Identical
  to NV on long/redundant contexts; diverges on short dense contexts for norm/similarity presses
  (same *quality*, different *tokens*). See *Limitations*.

---

## Usage

```bash
python -m sglang.launch_server --model-path <model> \
    --enable-kvpress \
    --kvpress-method knorm \
    --kvpress-compression-ratio 0.3 \
    --disable-radix-cache --disable-cuda-graph
```

| flag | meaning |
|---|---|
| `--enable-kvpress` | turn on compression |
| `--kvpress-method` | `knorm` (default) / `random` / `streamingllm` / `keydiff` / `lagkv` |
| `--kvpress-compression-ratio` | fraction of prefill tokens to drop (0.0–1.0; 0.3 = keep 70%) |
| `--kvpress-batched` | use stacked-gather + layer-vectorized score; same kept-set, ~10× fewer kernel launches |
| `--kvpress-per-head` | each KV head keeps its OWN top-`n_kept` tokens (J → 1.0 vs NV per-head); implies `--kvpress-batched` |

**Required companions** (asserted / auto-enforced in `Scheduler.__init__`): `--disable-radix-cache`,
`--disable-cuda-graph`, and the overlap scheduler is forced off. KVPress also rejects MLA/NSA models.
*Why each is required is explained under Limitations.*

---

## How it works

1. **Score (post-prefill).** For each request, gather its prefill K/V from the pool and compute a
   per-token importance score with the chosen press, **summed across all transformer layers** (the
   paged pool can keep only one token set for all layers, so per-layer scores are aggregated into one
   global ranking — see *Limitations / per-token*).
2. **Select & free.** Keep the top `(1 - ratio)` tokens (NVIDIA convention: *higher score = keep*);
   free the pruned slots back to the allocator.
3. **Compact.** Write the kept slots into `req_to_token[:n_kept]` (front-packed, gap-free) and record
   `req.actual_kv_len = n_kept` as the live **physical** length.
4. **Decode.** New tokens are written at the **physical** position (`actual_kv_len`, advanced each
   step) so kept-prefill + decode stay contiguous; `seq_lens` stays **logical** so RoPE positions
   remain correct; the attention backend reads the compacted range via `actual_kv_lens`.

Files: scoring in [`kvpress_methods.py`](kvpress_methods.py); engine integration in
`managers/scheduler.py` (`_kvpress_compress_single_req`), `mem_cache/common.py` (`alloc_for_decode`),
`managers/schedule_batch.py` (`actual_kv_lens` plumbing), `layers/attention/triton_backend.py` (read).

---

## Correctness: the bugs that were found and fixed

This integration started as an experimental draft that produced plausible-looking ROUGE numbers but
was, end-to-end, badly broken (compressed-vs-uncompressed quality ≈0.25–0.34 where the NV reference
was ≈0.98). Three independent bugs, all empirically confirmed:

1. **Score-sign inversion.** The scheduler negated every press's score before `topk`. That happened to
   be correct for `knorm` (which returned `+‖k‖` while NV returns `−‖k‖`, so two wrongs canceled) but
   **inverted** `keydiff`/`streamingllm`/`lagkv`, which already followed NV's "high = keep" convention
   — they kept exactly the tokens that should be pruned (kept-set Jaccard vs NV = **0.0** at ratio 0.5,
   the exact complement). Fix: `knorm` returns `−‖k‖`; remove the blanket negation; `topk(+score)`.
2. **Dead `actual_kv_lens` propagation (the dominant bug).** The physical compacted length was computed
   on `ScheduleBatch` and read on `ForwardBatch`, but **never copied through `ModelWorkerBatch`** in
   between, so the attention backend always saw `None` and fell back to the *logical* `seq_lens` — i.e.
   decode read into the freed zero-gap and missed every generated token. Even `knorm` (correct
   selection) cratered to 0.34 until this was wired through. Fix: add `actual_kv_lens` to
   `ModelWorkerBatch` + copy it in `get_model_worker_batch`.
3. **Layer-0-only selection.** The compaction looped over all layers but used only layer 0's scores
   (the rest were computed and discarded). `knorm`'s key-norm distribution varies sharply per layer, so
   layer-0 norms were a poor signal — `knorm` stalled at 0.42 even after (1)+(2). Fix: aggregate scores
   across **all** layers (cross-layer sum), recovering `knorm` to 0.98.

Plus: the decode write was going to the *logical* position (leaving a gap); the forced overlap-off
guard (the overlap scheduler speculatively allocated/freed decode slots and double-freed against the
compacted layout); and a large amount of leftover per-request `[KVPress Debug]` logging on hot paths
(some of it *unconditional*, hurting non-KVPress users) — all removed.

Validated four ways: a CPU score-alignment harness (per-token kept-set vs NV), a per-layer/per-head
analysis on real model K/V, a **real-engine tensor dump** (the running pool's K/V vs HF: max\|ΔK\|≈0.14
on 2.7M elements = fp16+RoPE noise; the engine's real kept set matches the offline prediction), and
end-to-end ROUGE parity vs NV (D~B ≈ 0.99).

---

## Limitations

### Per-token, not per-head (the architectural floor)

SGLang's KV pool is `k_buffer[layer] = [slots, kv_heads, head_dim]` with `req_to_token[req, pos] →
slot`: **one slot holds a token's KV for all heads**, and freeing is token-atomic. NVIDIA kvpress runs
on HF's dense per-head cache and prunes **each head independently** (different kept tokens per head).
SGLang physically cannot — you can't free "half a slot" — so it keeps a **single shared token set**
(the cross-layer-aggregated top-`n_kept`), which is the best a per-token method can do.

Consequence, measured: on **long / redundant** contexts and for **positional** presses
(`streamingllm`, `lagkv`) SGLang's kept set is indistinguishable from NV's (parity ≈ 0.99–1.0). On
**short / dense** contexts, the data-dependent presses (`knorm`, `keydiff`) diverge from NV's per-head
choices — the *quality* degrades equally (compression hurts both), but the *tokens kept* differ. This
is a property of the paged pool, not a bug.

> **Per-head packing IS implemented as an opt-in switch (`--kvpress-per-head`)**: each KV head keeps
> its own top-`n_kept` tokens at a uniform budget. SGLang stores K post-RoPE, so the physical row
> index is just a storage address — head `h` and head `h'` can independently hold different source
> tokens in their own column of row `r`, and the attention kernel (which reads each head's column with
> its own RoPE'd query) needs no change. Reaches NV per-head behavior at the same compression ratio
> and same memory. (*Variable* per-head budgets — AdaKV — are a different story: they would either
> negate the memory savings (pad-to-max) or require forking the whole memory subsystem; **not
> recommended**, see [design notes](../../../../../../../KVPRESS_DESIGN_NOTES.md).)

### Other constraints
- **RadixCache off:** a compressed prefix is request-specific and its slot→token map is renumbered, so
  it can't be shared/ref-counted in the radix tree. (A *suffix-only, consumer-not-producer* coexistence
  is feasible — see design notes — but not implemented.)
- **CUDA graph off:** per-request physical lengths are dynamic.
- **Overlap scheduler off:** the one-step look-ahead would alloc/free decode slots against the stale
  pre-compaction layout (double-free). Recoverable with an overlap-safe redesign (design notes §c).
- **Attention-free presses only:** SnapKV / ExpectedAttention / TOVA need observed attention scores,
  which SGLang's fused backends don't materialize. (`ExpectedAttention` is feasible via query
  statistics without kernel changes — a good future addition.)
- **Prefill-only:** decode-generated KV is not (yet) re-compressed, so the memory win decays for very
  long generations. (Decode-time re-press is a natural extension; the physical-write machinery already
  supports it.)

---

## Performance (compaction step, TinyLlama-class, num_valid=600, ratio=0.3)

Micro-benchmark of the post-prefill compression on the A800 (torch.profiler kernel counts):

| mode | flag | CUDA kernel launches | GPU µs | wall ms |
|---|---|---|---|---|
| current (per-layer loop) | default | 135 | 1172 | 1.752 |
| batched single-set | `--kvpress-batched` | **8** | 300 | **0.161** |
| per-head packing | `--kvpress-per-head` | 52 | 1074 | 0.790 |

`batched` is **same kept-set as default, ~11× faster wall** (kernel-launch-bound is collapsed). `per_head`
is ~2× faster than default *and* reaches NV per-head selection.

## Per-mode quality (3-mode e2e, ROUGE-L vs NV reference / vs full model)

Short dense prompt (DEFAULT_PROMPT, ratio=0.3, 64 new tokens) — the case that stresses
the single-set vs per-head difference:

| press | single → NV | batched → NV | **per_head → NV** | single → full | **per_head → full** |
|---|---|---|---|---|---|
| knorm | 0.458 | 0.458 | **0.652** | 0.449 | 0.465 |
| streamingllm | 0.997 | 0.997 | 0.997 | 0.541 | 0.541 |
| keydiff | 0.485 | 0.485 | 0.500 | 0.567 | **0.721** |
| lagkv | 0.997 | 0.997 | 0.997 | 0.541 | 0.541 |

Long redundant context (`The quick brown fox…`×40 — see `kvpress_e2e_parity.py`): all three modes
already reach **D~B ≈ 0.99** across presses (single-set is enough when redundancy is high). Per-head
matters when context is short and dense (the case above), where it raises the data-dependent presses
toward NV per-head behavior.

Caveats: streaming/lagkv don't differ between single and per-head because their score is positional
(all heads pick the same tokens); only knorm/keydiff benefit. `random` per_head selects different
tokens per head but quality is unchanged (random is a baseline floor).

## Roadmap (see [KVPRESS_DESIGN_NOTES.md](../../../../../../../KVPRESS_DESIGN_NOTES.md) for full analyses)

1. **Layer-importance-weighted aggregation** — replace the uniform cross-layer sum (magnitude-biased)
   with per-layer normalization + dispersion weighting. Free, directly raises the single-set ceiling.
2. **Memory-vs-quality benchmark frontier** (RULER / LongBench) — turn "aligned to NV" into
   "demonstrably useful": accuracy vs compression-ratio at fixed context length.
3. **Decode-time re-compression** — reuse the physical-write/live-read plumbing to bound long-gen memory.
4. **ExpectedAttention press** via query statistics (no kernel surgery).
5. **Compaction perf** — batch the per-layer gather into one kernel; make `actual_kv_lens` a single
   device-resident tensor (removes the per-decode `.item()` sync); optionally make KVPress overlap-safe.

---

## Reproducing the evaluation

Harnesses (in the project root, run inside the SGLang env):

| script | what it measures |
|---|---|
| `kvpress_align.py` | CPU: per-token score + kept-set vs NV (no GPU) |
| `kvpress_realvalue.py` | real engine pool K/V vs HF + engine kept-set vs NV-per-head |
| `kvpress_e2e_parity.py` | end-to-end greedy ROUGE-L: SGLang-compressed vs NV-compressed vs full |
| `kvpress_bench.py` | long-context passcode-recall **accuracy**: `full_model` / `nv_kvpress` / `sglang_kvpress` |
| `kvpress_logprob.py` | continuation **perplexity + KL** of the reference compressor vs full |

### Bigger-model results (Qwen2.5-7B-Instruct)

> Qwen2.5 is bf16-native; fp16 overflows to NaN in attention/MLP — run these in **bfloat16**.

**Distributional cost (HF, full vs NV-press; continuation perplexity + mean KL over a 1049-token
context).** This is the *reference compressor's* quality, the yardstick SGLang is aligned against:

| press | ratio | ppl_full | ppl_nv | KL(full‖nv) | top1_agree |
|---|---|---|---|---|---|
| knorm | 0.1 / 0.3 / 0.5 | 15.99 | 19.8 / 27.8 / 16.1 | 0.40 / 0.26 / 0.22 | 0.75–0.79 |
| keydiff | 0.1 / 0.3 / 0.5 | 15.99 | 14.7 / 22.0 / 25.4 | 0.51 / 0.25 / 0.55 | 0.67–0.75 |
| streamingllm | 0.1 / 0.3 / 0.5 | 15.99 | 359 / 247 / 402 | 3.8 / 2.6 / 3.2 | 0.29–0.38 |
| lagkv | 0.1 / 0.3 / 0.5 | 15.99 | 410 / 75 / 268 | 4.3 / 1.3 / 2.6 | 0.38–0.42 |

knorm/keydiff preserve the distribution (KL ≤ 0.55); **streamingllm/lagkv collapse** on this
repetitive context — positional presses drop the informative middle. (Their accuracy below tracks.)

**Long-context needle recall accuracy** (15 samples; `full_model` = no compression = 0.67):

| press | ratio | nv_kvpress (HF) | sglang_kvpress |
|---|---|---|---|
| knorm / keydiff / lagkv | 0.1 / 0.3 / 0.5 | 0.67 | 1.00 |
| streamingllm | 0.1 / 0.3 / 0.5 | 0.67 / 0.53 / 0.40 | 1.00 / 0.80 / 0.60 |

Clean within-engine signal: **NV's `streamingllm` recall falls with ratio** (0.67→0.40) while
`knorm`/`keydiff`/`lagkv` hold — matching the perplexity table. **Caveat:** `nv_kvpress` runs on HF and
`sglang_kvpress` on the SGLang engine, so their *absolute* accuracies are **not directly comparable**
(HF greedy underperforms "lost-in-the-middle" here; SGLang's generation is more robust, and
compression that keeps the distinctive needle can reduce distractors). The **rigorous** SGLang↔NV
alignment is the same-prompt end-to-end ROUGE parity (**D~B ≈ 0.99**, TinyLlama) and the real-engine
per-layer kept-set match — not this cross-engine accuracy, which is a directional quality sanity check.

_Harnesses: `kvpress_logprob.py` (perplexity/KL), `kvpress_bench.py` (needle accuracy)._
