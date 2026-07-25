- [Decoders](#decoders)
  - [1. Overview](#1-overview)
  - [2. CTC Decoders](#2-ctc-decoders)
  - [3. Transducer Greedy Decoders](#3-transducer-greedy-decoders)
  - [4. Transducer Beam Search (ALSD++)](#4-transducer-beam-search-alsd)
    - [4.1 Why alignment-length synchronous](#41-why-alignment-length-synchronous)
    - [4.2 The loop](#42-the-loop)
    - [4.3 Recombination](#43-recombination)
    - [4.4 The completed set `F`](#44-the-completed-set-f)
    - [4.5 Parameters](#45-parameters)
    - [4.6 Shallow fusion](#46-shallow-fusion)
    - [4.7 Usage](#47-usage)
    - [4.8 Deviations from the paper](#48-deviations-from-the-paper)
    - [4.9 Limitations](#49-limitations)
    - [4.10 Verification](#410-verification)
  - [5. References](#5-references)

# Decoders

## 1. Overview

Every model exposes two decoding entry points, both returning a `schemas.PredictOutput`:

| Method                                            | Purpose              |
| ------------------------------------------------- | -------------------- |
| `recognize(inputs, **kwargs)`                     | Greedy decoding      |
| `recognize_beam(inputs, beam_width=10, **kwargs)` | Beam search decoding |

`BaseModel.predict_step` calls **both** on every evaluation batch and emits `tokens` (greedy) plus `beam_tokens` (beam). `PredictLogger` writes them to a TSV as `path, groundtruth, greedy, beam`, which `app_util.evaluate_hypotheses` then scores into a `{"greedy": {...}, "beam": {...}}` report.

Both return `tokens` padded with the blank index, so blanks must be stripped by the tokenizer on detokenization. Transducer decoders additionally return `next_tokens`, `next_encoder_states` and `next_decoder_states` so decoding can be resumed on the next chunk of a stream; CTC decoders return `next_tokens=None`.

## 2. CTC Decoders

See [base_ctc.py](../tensorflow_asr/models/ctc/base_ctc.py).

Both are thin wrappers over TensorFlow builtins applied to the log-probability matrix:

- `recognize` → `tf.nn.ctc_greedy_decoder` with `merge_repeated=True`
- `recognize_beam` → `tf.nn.ctc_beam_search_decoder` with `beam_width`

There is no language model fusion on this path.

## 3. Transducer Greedy Decoders

See [base_transducer.py](../tensorflow_asr/models/transducer/base_transducer.py).

`recognize` dispatches on batch size:

- `recognize_single` (batch size 1) — `tf.TensorArray` based, honours `max_tokens_per_frame`, output width is `nframes * max_tokens_per_frame`.
- `recognize_batch` (batch size > 1) — fully batched `tf.while_loop`, output width is `max_frames * 2 + 1`.

Note that `recognize_batch` has **no** per-frame emission cap: on a frame where the argmax never selects blank it will keep emitting until the token budget runs out.

## 4. Transducer Beam Search (ALSD++)

Implemented in `Transducer.recognize_beam`, following **ALSD++** from Grigoryan et al., Interspeech 2025 \[1\], which is an optimized form of the alignment-length synchronous decoding of Saon et al. \[2\].

Notation used below: `B` batch, `W` beam width, `V` vocabulary, `T` encoder frames, `s` = `max_tokens_per_frame`, `t` = frames consumed by a hypothesis, `u` = labels emitted by a hypothesis.

### 4.1 Why alignment-length synchronous

A transducer hypothesis advances through a 2-D lattice: blank moves along the time axis, a label moves along the label axis. Every iteration of this decoder advances **each** hypothesis by exactly one lattice step, so all hypotheses in a beam always share the same alignment length `t + u`. Their accumulated log-probabilities therefore contain the same number of terms and can be compared directly by a single `top_k` — this is the defining property of ALSD, and the reason blank and label expansions can compete in one pool.

ALSD++ replaces ALSD's fixed `S = T + U_max` iteration budget with a frame-driven bound: iterate until every hypothesis has consumed all `T` frames, allowing at most `s` labels per frame. That cap is what stops a hypothesis that already reached the end of the audio from spending the leftover iterations on hallucinated tokens.

### 4.2 The loop

The beam is folded into the batch axis, so the prediction and joint networks run **once per step for all `B * W` hypotheses** — the "batch operations" of \[1\]. Per-hypothesis state carried through the `tf.while_loop`:

| Tensor           | Shape                                  | Meaning                             |
| ---------------- | -------------------------------------- | ----------------------------------- |
| `scores`         | `[B, W]`                               | accumulated log-probability         |
| `frame_indices`  | `[B, W]`                               | `t`                                 |
| `tokens`         | `[B, W, 2T+1]`                         | transcript                          |
| `tokens_length`  | `[B, W]`                               | `u`                                 |
| `num_expansions` | `[B, W]`                               | labels emitted on the current frame |
| `last_tokens`    | `[B, W]`                               | last non-blank label                |
| `states`         | `[B*W, num_rnns, nstates, state_size]` | prediction network states           |
| `hashes`         | `[B, W]`                               | rolling transcript hash             |

Each iteration:

1. Gather `encoded` at `min(t, T-1)` and run `call_next` for all `B * W` hypotheses → `log_probs [B, W, V]`.
2. **Mask.** A hypothesis at the expansion cap (`num_expansions >= s`, or `u` at the token budget) has its labels masked out but **still pays the real `ln p(blank)`** for consuming the frame. A hypothesis that consumed every frame (`t >= T`) is frozen instead: blank at log-probability `0`, so its final score is preserved while the rest of the beam finishes.
3. **Recombine** duplicates (see 4.3).
4. `candidates = scores[:, :, None] + log_probs` → `[B, W, V]`.
5. **Harvest** newly completed hypotheses into `F` (see 4.4).
6. `top_k` over the flattened `[B, W*V]` keeps the best `W`; `parent = index // V`, `token = index % V`.
7. Rebuild the beam by gathering on `parent`. Blank keeps the parent's prediction states and last token; a label takes the new states and appends the token.

Termination is `all(t >= T)`, with `maximum_iterations = T * (s + 1)` as the static bound — each hypothesis emits at most `s` labels before being forced to consume a frame.

Cost is one joint-network call on `B * W` rows per step, with at most `T * (s + 1)` steps.

### 4.3 Recombination

Two hypotheses spelling the same transcript differ only in blank placement and are the same hypothesis. \[1\] compares them in constant time via a rolling hash over the transcript:

```
H_{u+1} = (H_u * P + T_{u+1}) mod M      P = 1_000_003, M = 1_000_000_007
```

Because hypotheses are sorted by score after `top_k`, index order equals score order, so keeping the first occurrence and setting later duplicates to `-inf` keeps the best-scoring copy.

### 4.4 The completed set `F`

A hypothesis stops accumulating log-probabilities the moment it completes, while partial ones keep going. It is therefore **not** guaranteed to rank inside the top `W` at the step it completes, and pruning it there would lose it permanently.

Completed candidates are consequently harvested from the full `W * V` candidate set **before** the prune, not from the `W` survivors after it — this is what ALSD's final hypothesis set `F` means. A candidate is complete exactly when it takes the blank of the last frame; blank carries the parent's transcript, last token and states over unchanged, so the pre-prune tensors are already the right ones to record.

Only the best element of `F` is ever returned, so the implementation tracks a running argmax rather than materialising the whole set. Because a completed hypothesis has both its score and its length final at that moment, ranking it under the same criterion used to pick the winner keeps length normalization exact.

### 4.5 Parameters

```python
recognize_beam(inputs, beam_width=10, max_tokens_per_frame=3, score_norm=True, lm=None, lm_alpha=0.0)
```

| Parameter              | Default | Meaning                                                               |
| ---------------------- | ------- | --------------------------------------------------------------------- |
| `beam_width`           | `10`    | hypotheses kept per utterance (`W`)                                   |
| `max_tokens_per_frame` | `3`     | `s` in \[1\], max non-blank expansions on a single frame              |
| `score_norm`           | `True`  | divide the final score by transcript length before picking the winner |
| `lm`                   | `None`  | external language model to shallow fuse (see 4.6)                     |
| `lm_alpha`             | `0.0`   | `λ` of eq. (3) in \[1\], the fusion weight                            |

`score_norm` counteracts the bias of an accumulated log-probability towards short transcripts. Leaving it on is the usual choice; turning it off returns the maximum-probability path instead.

### 4.6 Shallow fusion

Equation (3) of \[1\] fuses an external LM into the per-step scores:

```
ln p_tot[k] = ln p[k] + λ·(ln(1 − p[∅]) + ln p_LM[k])    for k ≠ ∅
ln p_tot[∅] = (1 + λ)·ln p[∅]
```

The blank term is the part that matters. Boosting only the label scores would make blank comparatively cheaper at every frame and drive the deletion rate up, so blank is scaled by `(1 + λ)` to compensate. `ln(1 − p[∅])` is computed as `ln(−expm1(x))`, the stable form of `log1mexp`, clamped so a saturated `p[∅] = 1` cannot produce `ln(0)`.

Fusion is applied **before** the forced-blank masking of step 2, so a capped hypothesis pays the fused blank cost, and a completed one is still frozen at `0`.

**No language model ships with TensorFlowASR.** `recognize_beam` takes any object satisfying [`LanguageModel`](../tensorflow_asr/models/decoders/language_model.py):

```python
class LanguageModel(Layer):
    def get_initial_state(self, batch_size) -> tf.Tensor: ...  # [B, ...]
    def score(self, previous_tokens, previous_states): ...  # -> ([B, V], [B, ...])
```

Two constraints follow from where it is called:

- `score` runs **inside** the `tf.while_loop`, once per step on all `B * W` hypotheses. It must be pure TensorFlow — any python-side lookup breaks TFLite/XLA export.
- States must be a **single tensor**, batch on axis 0, any trailing rank. The beam re-orders them on axis 0 during recombination exactly as it does the prediction network states, so a python dict or ragged trie will not survive the loop.

`score` returns log-probabilities over the *transducer* vocabulary with matching indices; the blank column is never read.

### 4.7 Usage

```python
outputs = model.recognize_beam(inputs, beam_width=16, max_tokens_per_frame=2)
transcripts = model.tokenizer.detokenize(outputs.tokens)
```

TFLite export (see [tflite_convertion](./tutorials/tflite.md)):

```bash
tensorflow_asr tflite \
    --config-path=/path/to/config.yml.j2 \
    --h5=/path/to/weight.h5 \
    --bs=1 \
    --beam-width=16 \ # >0 enables ALSD++, 0 keeps greedy
    --output=/path/to/output.tflite
```

`make_tflite_function(batch_size, beam_width)` routes to `recognize_beam` whenever `beam_width > 0`, and to greedy otherwise.

Evaluation reads its settings from the tokenizer's `DecoderConfig`:

```yaml
decoder_config:
  beam_width: 16      # 0 (the shipped default) disables beam search
  norm_score: True    # -> score_norm
  lm_alpha: 0.5       # -> lm_alpha, eq. (3) lambda
  lm_config:          # keras serialization blob, built by model.make_lm()
    class_name: my_package>MyLanguageModel
    config: { ... }
```

`predict_step` calls `BaseModel.get_beam_decoding_kwargs()`, which returns nothing when `beam_width <= 0`. In that case the beam column of the evaluation output mirrors the greedy one rather than running a second decode — the same `beam_width > 0` convention `make_tflite_function` already uses. `model.make_lm()` builds the LM from `lm_config` and is called by `scripts/test.py` and `scripts/tflite.py`; it is a no-op when `lm_config` is empty.

The LM is deliberately **not** tracked as a keras sub-layer, so its weights never enter the ASR model's checkpoint.

### 4.8 Deviations from the paper

- **Transcript storage.** \[1\] uses a trie (`transcripts` + `transcripts_ptrs` backlinks) to avoid copying whole transcripts on each expansion. Here transcripts are dense `[B, W, 2T+1]` and re-gathered each step, because `tf.gather` over the beam axis is a single vectorized op and keeps every shape static, which is what XLA and TFLite export need. Memory is `O(B * W * T)`.
- **No bundled language model.** Equation (3) itself is implemented (4.6), but no concrete LM ships with the repository — \[1\] evaluates against an n-gram LM held on device. You supply the `LanguageModel`; the decoder only defines the interface and the fusion math.
- **CUDA graphs.** Not applicable — the loop is a `tf.while_loop`.

### 4.9 Limitations

- **Recombination is lossy when `s` binds.** Merging by transcript hash keeps the higher-scoring copy, which may be the one with *less* expansion budget left on the current frame. When `s` never binds this cannot happen and the search is exact.
- **Token budget.** `u` is capped at `max_frames * 2 + 1` to match the greedy batch decoder's output width. `max_frames` is the padded batch width, not the individual utterance length, so the effective cap for a short utterance in a mixed batch is looser than `2 * its own length + 1`.
- **`-inf` is `-1e9`**, kept finite so masked entries can be added to without producing `NaN`. Scores accumulate in `float32` regardless of the model's compute dtype.
- **LM state does not survive streaming.** `PredictOutput` carries `next_decoder_states` but has no field for LM state, so a fused LM restarts from `get_initial_state` on every chunk. Fusion is therefore only correct for whole-utterance decoding.
- **`lm_alpha=0` with an `lm` set still costs a full LM call per step** — it is a no-op mathematically, not computationally. Pass `lm=None` to skip the work.
- **No timestamps and no n-best output** — only the single best hypothesis is returned.

### 4.10 Verification

`tests/test_beam_search.py` drives the transducer with a stubbed joint network whose
log-probabilities depend only on `(frame, previous token)`. That keeps the search space small
enough to solve exactly with a dynamic program over the `(t, u, last, expansions)` lattice, so the
beam is compared against a **known optimum** rather than against itself. Run it with:

```bash
uv sync --extra dev && python -m pytest tests/test_beam_search.py
```

What is covered:

| Property                       | How                                                                                                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reproduces the DP optimum      | `s` non-binding + exhaustive beam, 3 lattice sizes x both `score_norm` modes                                                                                                 |
| Per-utterance correctness      | ragged batch (lengths 4/3/1) despite the batch-global loop bound                                                                                                             |
| Narrow beams stay legal        | `W` in 1/2/4/16, output length and token range checked                                                                                                                       |
| Pruning is the only loss       | width-1 misses what an exhaustive beam finds, and the exhaustive beam always hits the optimum                                                                                |
| Expansion cap holds            | transcript never exceeds the `s` budget under label-favouring distributions                                                                                                  |
| Graph == eager, determinism    | `tf.function` output compared to eager, repeated decodes compared to each other                                                                                              |
| Output contract                | shapes of `tokens` / `next_tokens` / `next_decoder_states`, no `NaN`                                                                                                         |
| Shallow fusion, eq. (3)        | fused objective vs the same DP at `λ` in 0 / 0.3 / 1.0 x both `score_norm` modes                                                                                             |
| `λ = 0` is a no-op             | a fused LM at `λ = 0` decodes identically to no LM                                                                                                                           |
| LM state threading             | the test LM's distribution depends on a state-carried emission count, so a beam that advanced LM state on a blank, or failed to re-order it onto the selected parents, fails |
| Config plumbing                | `get_beam_decoding_kwargs` for `beam_width <= 0`, `norm_score`, and LM attachment                                                                                            |
| LM stays out of the checkpoint | attaching an LM does not change `model.weights`                                                                                                                              |

Beyond the suite, TFLite conversion of `recognize_beam` succeeds under `jit_compile=True`.

Two caveats on what this does **not** establish:

- There is no trained checkpoint in this repository, so these are correctness results against a known optimum, **not** WER measurements. In particular no claim is made that shallow fusion improves accuracy on real audio — only that it computes eq. (3) correctly.
- With a binding `s` and a narrow beam — the actual operating regime — exactness is not guaranteed and is not asserted; pruning and transcript-hash recombination are both lossy by design.

## 5. References

1. L. Grigoryan, V. Bataev, A. Andrusenko, H. Xu, V. Lavrukhin, B. Ginsburg. *Pushing the Limits of Beam Search Decoding for Transducer-based ASR models*. Interspeech 2025. <https://arxiv.org/abs/2506.00185>
2. G. Saon, Z. Tüske, K. Audhkhasi. *Alignment-Length Synchronous Decoding for RNN Transducer*. ICASSP 2020.
3. A. Graves. *Sequence Transduction with Recurrent Neural Networks*. 2012. <https://arxiv.org/abs/1211.3711>
4. V. Bataev, H. Xu, D. Galvez, V. Lavrukhin, B. Ginsburg. *Label-Looping: Highly Efficient Decoding for Transducers*. Interspeech 2024. <https://arxiv.org/abs/2406.06220>
