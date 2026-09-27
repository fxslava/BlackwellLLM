# LongBench baseline — GLM-4-9B E8W5 (5.14 bpw lattice)

The reference point for every later weight-side optimization (Hessian-weighted
error feedback, Hadamard/rotation preprocessing, the MMA-permuted pack). It is a
**quality + latency + VRAM baseline measured end to end through the engine**, not a
kernel microbenchmark: the numbers come from `blackwell_llm.exe` loading the
converted checkpoint and answering LongBench prompts through the COM boundary.

Companion documents: [`E8W5_FORMAT_SPEC.md`](E8W5_FORMAT_SPEC.md) (the format),
[`E8W5_BLACKWELLLLM_INTEGRATION.md`](E8W5_BLACKWELLLLM_INTEGRATION.md) (the engine
wiring, the kernel throughput, and the §6 register this document closes rows 12
and 13 of).

---

## 0. Read this before quoting a number

**The scores below are truncation-bound, not quantization-bound.** Prompts were
middle-truncated to a **2000-token budget**, while these four tasks average
3.5k–7.5k tokens. For `qasper` and `multifieldqa_en` that removes most of the
document the question is about, so their scores measure *what the model can do
with a fragment of the evidence*. They are a valid **internal A/B reference** —
run the same harness after an optimization pass and the delta is meaningful,
because the truncation is identical — and they are **not comparable to published
LongBench numbers** for any model.

The budget was a deliberate GPU-time choice, not a limitation of the harness: the
same command with `--max-input-tokens 16000` truncates only 3 of the 200 samples
and costs ~7.0 h instead of the 1.7 h this run took. §6 has the cost table.

Concretely, what truncation cost: 18 of 50 `qasper` samples answered
`"unanswerable"` because the evidence had been cut, each scoring 0.000. Excluding
those, the remaining 32 average **40.60 F1 against a full-precision published
43.3** — see §3. `multifieldqa_en`, `lcc` and `trec` land at or above their
published full-precision counterparts, so the format itself is not what `qasper`'s
number is measuring.

---

## 1. Reproducing this run

```bash
python benchmarks/run_longbench_harness.py \
    --model F:/AI/models/GLM-4-9B-e8w5 \
    --tasks qasper,multifieldqa_en,lcc,trec \
    --limit 50 --max-input-tokens 2000 --ctx 4096 \
    --kv-mode paged --prefill fast \
    --output-dir benchmarks/results/e8w5_baseline
```

The harness downloads LongBench's `data.zip` from the HF hub on first use and
caches the four task files under the gitignored `benchmarks/data/`. Render the
tables below from the saved run, or diff a later run against it:

```bash
python benchmarks/report_tables.py benchmarks/results/e8w5_baseline
python benchmarks/report_tables.py benchmarks/results/<new-run> --against benchmarks/results/e8w5_baseline
```

**Binary provenance matters for the "after" side of any A/B.** This run was measured
on an x64-Release build of `e1cc792` plus this session's uncommitted prefill work
(`prefill_status` / `PrefillTokens` / the `--serve` flags), built before a parallel
attention-kernel (Split-K) session began editing `attention.cu`, `memory_pool`,
`runtime_config` and `continuous_kv_manager` in the same tree. The running process
held its loaded DLL image throughout, so none of those edits are in these numbers —
which is what makes this a clean **pre-Split-K** baseline. The full timeline, and the
list of files that are deliberately *not* in the measured binary, is in
`benchmarks/results/e8w5_baseline/SUMMARY.txt`. Rebuild and record provenance the same
way before measuring against it.

Everything needed to re-score without re-running inference is in
`benchmarks/results/e8w5_baseline/`: `predictions.jsonl` (one row per sample —
prediction, gold answers, token counts, per-phase latency, scores),
`summary.json`, and `engine.log` (the engine's own arena/DirectStorage output).

---

## 2. Method, and where it departs from the official script

Taken verbatim from THUDM/LongBench's `pred.py` and `config/*.json`:

| | |
|---|---|
| prompt templates | the official `dataset2prompt` string for each of the four tasks |
| generation length | official `dataset2maxlen`: `qasper` 128, the other three 64 |
| chat template | applied to `qasper` / `multifieldqa_en`; **not** to `lcc` / `trec`, matching the official exclusion list |
| truncation | middle-truncation (keep the head and the tail, drop the middle) |
| scoring | `qa_f1` for both QA tasks, `code_sim` for `lcc`, `classification` for `trec`; `max` over multiple gold answers; first-line-only postprocessing for `trec` |
| sample selection | the first N rows of each task file, in file order |

Three deliberate differences, each of which could move a score:

1. **Truncation runs on token ids inside the engine**, against the engine's own
   tokenizer, rather than in Python against a second tokenizer. The `--serve`
   protocol takes a token budget and the engine applies it; the document is
   truncated *before* the chat template is applied, so the template's framing and
   the assistant cue always survive. This removes a whole class of
   "my tokenizer disagreed with the one doing inference" error.
2. **Greedy decoding** (`temperature = 0`, engine-side argmax). The official
   script uses sampling for some models; greedy makes the baseline
   bit-reproducible, which is the property that matters for a diff.
3. **Edit similarity is the Indel/Levenshtein ratio** (`2·LCS/(len_a+len_b)`,
   what `rapidfuzz.fuzz.ratio` and `python-Levenshtein.ratio` compute). The
   official script calls `fuzzywuzzy.fuzz.ratio`, which is *either* that or
   `difflib`'s Ratcliff–Obershelp ratio depending on whether the C extension is
   installed. `benchmarks/metrics.py` implements both and records the gap — on
   this run it is small (§5), but it is not zero, so the metric is stated rather
   than assumed.

`benchmarks/metrics.py` has no third-party dependencies (this machine has neither
`fuzzywuzzy`/`rapidfuzz` nor `rouge`) and self-tests against hand-computed values
with `python benchmarks/metrics.py`. A baseline that cannot be re-scored because a
package is missing is not a baseline.

---

## 3. Results

Run of 2026-09-27. 200/200 samples scored, **zero failures**, zero engine restarts,
102.6 min wall clock. Greedy, so re-running the same command reproduces these rows.

| task | metric | n | score | +/- stderr | mean prompt tok | truncated | s/sample |
|---|---|---:|---:|---:|---:|---:|---:|
| `qasper` | qa_f1 | 50 | **25.98** | 5.36 | 2019 | 100% | 28.3 |
| `multifieldqa_en` | qa_f1 | 50 | **50.26** | 4.93 | 2007 | 94% | 29.2 |
| `lcc` | edit_sim | 50 | **59.90** | 4.55 | 1870 | 66% | 32.7 |
| `trec` | classification | 50 | **81.00** | 5.51 | 2002 | 100% | 32.9 |

The stderr column is not decoration. On 50 samples it is 4.5–5.5 points, so **any later
run that moves a task average by less than ~10 points has moved it by noise**. Use
`report_tables.py --against` for a comparison: it pairs the identical samples and
reports a paired t, which resolves far smaller real differences than these
independent error bars suggest.

Secondary metrics, recorded per sample for diagnosis (not the official score):

| task | secondary metric | value |
|---|---|---:|
| `qasper` | rouge_l | 25.82 |
| `multifieldqa_en` | rouge_l | 49.75 |
| `lcc` | exact_prefix | 24.00 |
| `lcc` | edit_sim_difflib | 58.11 |

`lcc`'s two edit-similarity implementations differ by **1.79 points** (59.90 Indel vs
58.11 Ratcliff-Obershelp) — the ambiguity in the official script's `fuzz.ratio`, §2
item 3, made concrete. Quote the metric, not just the number.

### Raw vs refusal-excluded

`qasper`'s 25.98 is not a quantization result; it is a refusal result. 18 of the 50
samples answered `"unanswerable"` where the gold answer was a real one — the
truncation removed the evidence — and every one of those scores exactly 0.000.
Excluding them:

| `qasper` view | n | qa_f1 |
|---|---:|---:|
| all samples (the official score) | 50 | **25.98** |
| spurious refusals, excluded below | 18 | 0.00 |
| conditional on attempting an answer | 32 | **40.60** |
| _published ChatGLM3-6B-32k, full precision, full context_ | _—_ | _43.3_ |

So on the samples where 2k of context was enough to attempt an answer at all, the
5.14 bpw model scores **40.60 against a full-precision 43.3** at 16–32k of context.
That is the strongest statement this run supports about the format's quality, and it
is the number to watch after a Hessian or Hadamard pass.

---

## 4. Latency and VRAM

Per-task, from the engine's own instrumentation (prefill and decode timed
separately, after the DirectStorage load — closing row 13 of the integration doc's
register):

| task | prefill ms/token | mean prefill | decode ms/token | stop reasons |
|---|---:|---:|---:|---|
| `qasper` | 13.75 | 27.8 s | 63.5 | eos=50 |
| `multifieldqa_en` | 14.02 | 28.1 s | 64.5 | eos=49, limit=1 |
| `lcc` | 15.37 | 28.4 s | 68.1 | limit=50 |
| `trec` | 14.30 | 28.6 s | 66.2 | limit=50 |

**Prefill 13.75–15.37 ms/token (~65–73 tok/s). Decode 63.5–68.1 ms/token (~15.6
tok/s).** Decode costs **4.6x more per token than prefill**, which is the headline
performance finding of this run.

Part of that gap is expected and by construction: prefill skips the 1.24 GiB
`lm_head` read for every interior token and runs attention/norms/RoPE batched over a
64-token tile, while decode pays the head and the sampler on every single token. What
is *not* explained by construction is the size of it — 50 ms/token of unattributed
cost. Candidate mechanisms, in the order worth testing:

- **Paged-mode decode overhead.** This run used `--kv-mode paged` because that is what
  raises the token capacity and enables chunked prefill. Paged decode resolves a block
  table per step, where Continuous indexes a flat buffer. If Paged decode is
  materially slower, the engine's *real* workload (short prompt, long generation —
  the hotkey translator) may want the opposite mode from a benchmark harness.
- **The sampler's per-token allocation.** `sample_top_p`'s greedy branch
  (`src/kernels/sampling.cu`) does a `cudaMalloc` + argmax launch + 4-byte D2H
  `cudaMemcpy` + `cudaFree` **per token**. Both the allocation pair and the D2H copy
  are synchronising, on a path the engine's own comments elsewhere forbid
  (`d_next_token` in `step_embedding` exists precisely to avoid a per-step
  malloc/free). This is the cheapest thing to fix and the easiest to measure.
- **`lm_head` at batch 1.** 1.24 GiB per token at the ~145 GB/s this format achieves
  is ~8.6 ms, i.e. ~13% of a decode step — real but not the bulk.

**None of these is confirmed.** The A/B that separates the first from the rest —
identical prompt and context, `--kv-mode continuous` vs `paged`, comparing
`decode_ms/gen_tokens` — was written (`--serve` reports both phases, so it is a
two-minute run) but cancelled before it produced data when GPU profiling was
suspended. Treat the three bullets as ranked hypotheses, not findings.

### VRAM

| | |
|---|---|
| weights, contiguous arena | **7.231 GB** (from the engine's own arena log) |
| KV + dynamic pool, ctx 4096 Paged | **646.7 MB** (~158 KB/token) |
| GPU total in use, before load | 936 MB (desktop) |
| GPU total in use, peak during run | **11 846 MB** of 12 227 MB |
| peak over baseline | 10 910 MB |
| engine restarts / failed samples | 0 / 0 |

Two things to carry forward. First, **the peak sits 381 MB under the card limit** even
though weights + KV account for only 7.88 GB — roughly 3 GB goes to staging,
workspaces and logits, so "weights fit in VRAM" is not the same as "the run fits in
VRAM". Second, per-process VRAM is unavailable on this platform (WDDM reports N/A per
compute app), so the totals above are whole-device and include whatever the desktop
held; the arena figures are the authoritative per-engine numbers, and that is why the
harness parses them out of the engine log.

---

## 5. Failure modes

#### `qasper` — truncation becomes refusal (the dominant failure mode)

| | n |
|---|---:|
| predicted `"unanswerable"` | 20/50 |
| …of those, gold was also unanswerable (correct refusal) | 2 |
| …spurious refusals, all scoring 0.000 | **18** |
| exact zeros, all causes | 28/50 |
| perfect 1.000 | 7/50 |
| mean generated tokens | 8.8 |

The model is behaving *correctly per its instructions*: the official `qasper` prompt
says to write `"unanswerable"` when the article does not contain the answer, and after
middle-truncation to 2000 tokens it usually does not. This is a **harness budget
artifact, not a model or quantization defect** — and it is concentrated, not diffuse:
18 of the 28 zeros are this one behaviour. It is also the single strongest argument
for re-running at `--max-input-tokens 16000` (§8).

#### `lcc` — the official first-line rule dominates the score

The official `code_sim_score` takes the first predicted line containing no `` ` ``, `#`
or `//`. On this run that rule fires hard:

| | n | edit_sim |
|---|---:|---:|
| rule scored line 1 (no fence, no inline comment) | 25 | **77.48** |
| rule skipped line 1 (fence or inline comment present) | 25 | **42.32** |
| _if the raw first line were scored instead_ | _50_ | _45.72_ |
| official score (the rule as written) | 50 | **59.90** |

Two distinct behaviours collide here, pulling in opposite directions:

- **Markdown fences.** 20/50 predictions open with ```` ```csharp ```` / ```` ```python ````.
  GLM-4-9B-Chat is a chat model, and `lcc` is one of the tasks the official protocol
  runs *without* a chat template — so the fence is the chat habit leaking into a raw
  completion. Here the rule **helps**: skipping the fence finds the real code line.
- **Inline comments.** When the model writes `foo(bar)  # Step 1: ...`, the `#` test
  discards a line that was otherwise an exact match, and scoring falls to the second
  or later line. Here the rule **hurts**.

Net, the rule is worth +14.18 points over scoring the raw first line, so it is doing
more good than harm — but half the samples are not scored on the line the model
actually predicted, and `exact_prefix` is only 24%. A future pass that changes the
model's commenting habits will move this score without changing its code quality.

One counter-intuitive observation, recorded without over-reading it: truncated `lcc`
samples score **higher** than untruncated ones (62.86, n=33 vs 54.15, n=17). Next-line
prediction depends on nearby context, which middle-truncation preserves at both ends;
the untruncated samples are also the shorter, more varied files. 17 samples is too few
to call this.

#### `trec` — no EOS, so latency is fixed by `max_gen`

All 50 samples stop at `limit` with exactly 64 generated tokens: raw-completion mode
has no stop token to hit, so every sample pays the full `dataset2maxlen`. The official
scorer only reads the first line, so the remaining ~60 tokens are pure cost — roughly
4 s per sample, ~3.5 min of this run. Scores land only on {0.0, 0.5, 1.0}, which is the
substring-voting rule: 0.5 means the prediction named the gold class and one other.

#### `multifieldqa_en` — the task that survived truncation

6 zeros, 10 perfect, and only 3/50 samples untruncated, yet **50.26 against a
published 51.7**. Its questions are answerable from a single field of the document,
which middle-truncation is much more likely to leave intact than `qasper`'s
scattered-evidence questions. This is the cleanest task in the set for A/B purposes:
high score, low refusal rate, and truncation-insensitive.

---

## 6. The prefill cost model, and the change that moved it

Prompt prefill, not decode, dominates a long-context benchmark: a 2000-token
prompt answered in 10 tokens spends >99% of its time in prefill. The engine had
no prefill entry point — callers looped `Forward()` once per token — so this work
added one.

### 6.1 `PrefillTokens` / `prefill_status`

`IBlackwellEngine::PrefillTokens` (and the facade's `prefill_status`) consumes a
whole prompt and samples only the continuation after its last token. Two things
make it cheaper than the caller's loop, and neither changes the result:

- **It skips the `lm_head` for discarded logits.** A `Forward()` loop runs the
  final RMSNorm and the `lm_head` GEMV for *every* prompt token; at
  `vocab_size = 151552`, `hidden = 4096`, BF16 that head is a **1.24 GiB read per
  token**, and the caller throws away all but the last result.
- **It tiles the prompt through the engine's batched chunk path** (`Impl::run_chunk`)
  wherever the resolved token capacity allows — Paged mode on a dense model, where
  `resolve_token_capacity()` returns 64. Under Continuous mode the capacity is 1
  and it falls back to the per-token sweep, so the call is always safe: it is
  deliberately *not* capability-gated, because skipping work is never an error.

For E8W5 specifically the chunk path does **not** batch the linear projections —
per [`E8W5_BLACKWELLLLM_INTEGRATION.md`](E8W5_BLACKWELLLLM_INTEGRATION.md) §5 the
obstruction is structural (an `mma.sync` B-fragment cannot hand one lane all eight
E8 coordinates, so a batched launch would decode each block four times), and the
format falls through to the per-row GEMV sweep. What chunking *does* batch is the
attention, the norms, the RoPE and the KV append — and that turns out to be where
the context-dependent cost lived.

### 6.2 Measured cost, both paths

GLM-4-9B E8W5, RTX 5070, x64-Release, greedy. Fitted on measured
`(prompt_tokens, prefill_ms)` pairs reported by the engine itself:

| path | fitted model | fitted over | quadratic coefficient |
|---|---|---|---|
| `continuous/loop` (one `Forward()` per prompt token) | `17.2·N + 6.43e-3·N²` ms | N = 348…2550, ±1.5% | 1× |
| `paged/fast` (`PrefillTokens`) | `15.6·N + 6.26e-4·N²` ms | N = 1983…9142, ctx 16384 | **21× smaller** |

Per-prompt prefill time, and what that implies for a 200-sample run:

| prompt tokens | `continuous/loop` | `paged/fast` | speedup |
|---:|---:|---:|---:|
| 2 000 | 60 s | 34 s | 1.8× |
| 4 000 | 172 s | 73 s | 2.4× |
| 8 000 | 549 s | 165 s | 3.3× |
| 16 000 | 1 921 s (32 min) | 410 s (6.8 min) | **4.7×** |

| input budget | samples truncated | est. run time, `paged/fast` | `continuous/loop` |
|---:|---:|---:|---:|
| 2 000 (this run) | 181/200 | **1.8 h** | 3.3 h |
| 4 000 | 131/200 | 3.5 h | 8.1 h |
| 8 000 | 58/200 | 5.9 h | 17.6 h |
| 16 000 | 3/200 | 7.0 h | 23.3 h |

**The projection was checked against the run it predicted.** For the 2000-token row
the harness projected 1.8 h and 181/200 truncated samples before starting; the run
took **1.71 h** and truncated **180/200**. The table's other rows use the same fitted
model and the same character-based token estimate, so treat them as good to ~10%.

The quadratic term survives in both paths — it is the attention reading a KV cache
that grows with position — so "flat ms/token" is *not* what this buys. At 1.3k
tokens the fast path looked flat at ~13.5 ms/token; extending the measurement to
9.1k showed it rising to 21.3 ms/token. Projections in the harness use the fitted
model, and the harness prints its projection before holding the GPU.

### 6.3 Parity

`PrefillTokens` must not change what the model says. Checked by running the same
prompts through both paths in the same KV mode and comparing the generated text
byte for byte:

| comparison | parity | prefill speedup |
|---|---|---|
| `paged/fast` vs `paged/loop` | **3/3 byte-identical** | 1.43× / 1.77× / 2.39× (N = 166 / 563 / 1327) |
| `continuous/fast` vs `continuous/loop` | **3/3 byte-identical** | 1.12× / 1.09× / 1.08× |
| `paged/fast` vs `continuous/loop` | 2/3 — one token differs mid-generation | — |

The last row is a **KV-mode** difference, not a prefill-path one: Continuous and
Paged run different attention kernels, and under greedy decoding a small logit
difference flips one token and the continuation diverges from there. Within a mode
the two prefill paths agree exactly. `--prefill loop` exists in `--serve` precisely
so this check is re-runnable.

---

## 7. Comparison with full-precision numbers

### 7.1 Published LongBench scores (context, not a baseline)

Official LongBench results, each model at **its own native context length** and in
**full precision** — the closest relative to GLM-4-9B by lineage is ChatGLM3-6B-32k:

| model | Qasper | MultiFieldQA-en | LCC | TREC |
|---|---:|---:|---:|---:|
| GPT-3.5-Turbo-16k | 43.3 | 52.3 | 54.7 | 68.0 |
| ChatGLM3-6B-32k | 43.3 | 51.7 | 57.66 | 79.0 |
| ChatGLM2-6B-32k | 31.5 | 46.2 | 55.6 | 62.5 |
| LongChat-v1.5-7B-32k | 27.7 | 41.4 | 53.0 | 63.5 |
| Llama2-7B-chat-4k | 19.2 | 36.8 | 52.4 | 61.5 |

Source: [THUDM/LongBench README](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md).

**Do not read a quantization loss out of the gap between these and §3.** Three
confounds stack: a different model, full precision, and — dominating both — 16–32k
of context against this run's 2k. On `qasper` the published models see the whole
article; this run sees a third of one.

### 7.2 Why there is no same-machine BF16 row

The honest comparison is the *same* checkpoint in BF16 through the *same* harness.
It is not affordable on this card: `GLM-4-9B-Chat-1M-hf` is 18 GB of BF16 weights
against 12 GB of VRAM, so it runs only with `--gpu-layers 17`, streaming 23 of 40
layers from pinned host RAM every token. That is ~9.5 GB of PCIe traffic per token
against E8W5's fully-resident 7.23 GB, and it makes a 200-sample LongBench run a
multi-day job rather than the 1.7-hour one this run took.

What exists today instead is the 83-token perplexity A/B in
[`E8W5_BLACKWELLLLM_INTEGRATION.md`](E8W5_BLACKWELLLLM_INTEGRATION.md) §4.5
(12.4701 → 13.0102, +4.33%), which that document is explicit about not being a
perplexity result. Closing this gap properly wants either a smaller paired subset
(BF16 at a 500-token budget over ~20 samples is roughly an hour) or a second GPU.

---

## 8. What to do next

1. **Re-run at a larger budget when there is GPU time.** `--max-input-tokens 16000`
   truncates 3/200 samples and costs ~7 h. That run is citable; this one is only
   self-comparable.
2. **The MMA-permuted pack** (§3 of the integration doc's next-steps list) is what
   would make prefill fast rather than merely faster: it unlocks a batched E8W5
   GEMM, and the linear projections are what the chunk path still cannot batch.
3. **A paired BF16 subset** at a small budget, to put a number on the quality cost
   of 5.14 bpw through this harness rather than through 83 tokens of perplexity.
4. **Watch `trec`'s `stop=limit` rate** (§5): the raw-completion tasks have no EOS
   to stop on and always run to `max_gen`, so their latency is fixed by
   `dataset2maxlen`, not by the model.

---

## 9. Register rows this closes

Updates [`E8W5_BLACKWELLLLM_INTEGRATION.md`](E8W5_BLACKWELLLLM_INTEGRATION.md) §6:

| # | Claim | Was | Now |
|---|---|---|---|
| 12 | Prefill / batched throughput | NOT measured (no batched kernel) | **measured**, both paths, §6.2 — still no batched E8W5 GEMM, but the chunked path's 21× smaller quadratic term is quantified |
| 13 | End-to-end tok/s isolated from load time | NOT measured | **measured**, §4 — the harness reports prefill and decode separately per sample, after load |
