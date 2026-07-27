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
    - [4.7 Internal LM subtraction: ILME and LODR](#47-internal-lm-subtraction-ilme-and-lodr)
    - [4.8 Usage](#48-usage)
    - [4.9 Deviations from the paper](#49-deviations-from-the-paper)
    - [4.10 Limitations](#410-limitations)
    - [4.11 Verification](#411-verification)
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

There is no language model fusion on this path. Worth knowing why the rest of this page does not apply here either: CTC assumes the output tokens are conditionally independent given the audio, so it has no label-history path and therefore **no internal language model to subtract**. The correction of 4.7 is a transducer-only concern. Fusing an *external* LM into CTC is a different construction again — the standard one is prefix beam search scoring `p_net(W|X) · p_LM(W)^α · |W|^β` at word boundaries \[7\] — and is not implemented here.

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
recognize_beam(inputs, beam_width=10, max_tokens_per_frame=3, score_norm=True,
               lm=None, lm_alpha=0.0, lm_type="shallow", internal_lm=None, lm_beta=0.0)
```

| Parameter              | Default     | Meaning                                                                  |
| ---------------------- | ----------- | ------------------------------------------------------------------------ |
| `beam_width`           | `10`        | hypotheses kept per utterance (`W`)                                      |
| `max_tokens_per_frame` | `3`         | `s` in \[1\], max non-blank expansions on a single frame                 |
| `score_norm`           | `True`      | divide the final score by transcript length before picking the winner    |
| `lm`                   | `None`      | external language model to shallow fuse (see 4.6)                        |
| `lm_alpha`             | `0.0`       | `λ` of eq. (3) in \[1\], the external fusion weight                      |
| `lm_type`              | `"shallow"` | internal LM correction: `"shallow"` (none), `"ilme"`, `"lodr"` (see 4.7) |
| `internal_lm`          | `None`      | the low-order LM that `"lodr"` subtracts                                 |
| `lm_beta`              | `0.0`       | `λ_I` of eq. (27) in \[5\], the internal LM weight                       |

In a config file `lm_type`, `lm_alpha` and `lm_beta` come straight from `decoder_config` under those names; `lm` and `internal_lm` are built from `lm_config.external_config` and `lm_config.internal_config` (4.8).

`score_norm` counteracts the bias of an accumulated log-probability towards short transcripts. Leaving it on is the usual choice; turning it off returns the maximum-probability path instead.

### 4.6 Shallow fusion

Equation (3) of \[1\] fuses an external LM into the per-step scores:

```
ln p_tot[k] = ln p[k] + λ·(ln(1 − p[∅]) + ln p_LM[k])    for k ≠ ∅
ln p_tot[∅] = (1 + λ)·ln p[∅]
```

The blank term is the part that matters. Boosting only the label scores would make blank comparatively cheaper at every frame and drive the deletion rate up, so blank is scaled by `(1 + λ)` to compensate. `ln(1 − p[∅])` is computed as `ln(−expm1(x))`, the stable form of `log1mexp`, clamped so a saturated `p[∅] = 1` cannot produce `ln(0)`.

Fusion is applied **before** the forced-blank masking of step 2, so a capped hypothesis pays the fused blank cost, and a completed one is still frozen at `0`.

`recognize_beam` takes any object satisfying [`LanguageModel`](../tensorflow_asr/models/lm/language_model.py):

```python
class LanguageModel(keras.Model):
    def get_initial_state(self, batch_size) -> tf.Tensor: ...   # [B, ...]
    def call(self, tokens, training=False): ...                 # [B, U]      -> [B, U, V]
    def call_next(self, previous_tokens, previous_states): ...  # ([B,1], [B,...]) -> ([B,V], [B,...])
```

It is a full `keras.Model`, not a layer, because it is trained on its own by `train_lm` and saved to its own h5. There are **two** forward passes because training and decoding want different shapes:

- `call` is **training** — teacher forced over a whole sequence, which is what `fit` runs, so it is the keras-idiomatic `call`.
- `call_next` is **decoding** — one step, carrying state. The beam search calls this. The name is the one the rest of the repository already uses for a single stateful step (`TransducerPrediction.call_next`, `Encoder.call_next`, `Transducer.call_next`).

Three constraints follow from where `call_next` runs:

- It runs **inside** the `tf.while_loop`, once per step on all `B * W` hypotheses. It must be pure TensorFlow — any python-side lookup breaks TFLite/XLA export.
- Weights must exist **before** decoding. `make()` builds them; a model first built inside the loop body would be creating variables inside a `tf.while_loop`, which graph mode rejects.
- States must be a **single tensor**, batch on axis 0, any trailing rank. The beam re-orders them on axis 0 during recombination exactly as it does the prediction network states, so a python dict or ragged trie will not survive the loop.

Both return log-probabilities over the *transducer* vocabulary with matching indices; the blank column is never read by the decoder.

Two token conventions are shared with the decoder. The **blank index doubles as start of sentence** — the beam holds `last_token = blank` until a hypothesis emits its first label — so training feeds `[blank, t1, ..., t_{n-1}]` to predict `[t1, ..., tn]` (`shift_tokens` does this). And there is **no end-of-sentence symbol**: a hypothesis ends when the frames run out, never on an emitted token.

One implementation ships: [`LSTMLanguageModel`](../tensorflow_asr/models/lm/lstm_language_model.py). Its defaults reproduce the external LM of the ILME paper \[5\] — two 2048-unit LSTM layers over a 512-dimensional embedding with input and output embeddings tied, which is 58M parameters at their 3999 word-pieces:

```yaml
lm_config:
  external_config:
    class_name: tensorflow_asr.models.lm.lstm_language_model>LSTMLanguageModel
    config:
      vocab_size: 1000
      embed_dim: 512
      units: 2048
      nlayers: 2
      tie_embeddings: True
```

Tying is why there is a projection layer: reusing the `[V, E]` embedding matrix as the output layer needs an `E`-wide input, but the last LSTM is `units`-wide, so a `units → embed_dim` projection sits between them. `tie_embeddings: false` drops it for an ordinary `units → V` dense layer, which is bigger and usually slightly worse.

**Start much smaller than the defaults.** At 58M parameters this runs once per beam step on all `B * W` hypotheses, and transducer decoding is hundreds of small sequential steps per utterance — the LM will dominate the runtime. `units: 512, nlayers: 2` is a reasonable first try; grow it only if the WER pays for the time.

### Where the pretrained weights aren't

There is no drop-in checkpoint, and the reason is worth stating once. The k2/icefall LODR recipe does publish one ([`ezerhouni/icefall-librispeech-rnn-lm`](https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm), a 3-layer 2048-unit RNN over BPE 500), but `call_next` returns log-probabilities indexed against **your** transducer's vocabulary. Using those weights would mean adopting icefall's exact SentencePiece model — same merges, same integer indices — for the ASR model too, on top of converting PyTorch weights to Keras. Retraining is the cheaper path, which is what `train_lm` is for.

What *is* reusable is the corpus. ILME and LODR both train on the LibriSpeech LM corpus, ~800M words against the ~9M words of LibriSpeech transcripts, and it is a single download:

```bash
wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
tensorflow_asr train_lm ... --target=external --text-path=librispeech-lm-norm.txt.gz \
    --output=lm.weights.h5
```

`.gz` is read directly and the file is streamed, so the ~4 GB decompressed size never has to be materialised. `--max-lines` caps it for a quick run.

One number not to copy blindly: the icefall recipe recommends `--lm-scale 0.42` with `--ngram-lm-scale -0.24`. That scale is **negative** because icefall *adds* it, in the density-ratio form. This repository *subtracts*, so the equivalent is `lm_alpha: 0.42` with `lm_beta: 0.24` — positive. Carrying the minus sign across would double the internal LM instead of removing it, and would decode perfectly happily while doing so.

### 4.7 Internal LM subtraction: ILME and LODR

Shallow fusion has a known weakness. A transducer is trained on paired speech and text, and in doing so it learns a language model of its training transcripts whether anyone asked for one or not — this is the **internal language model**. When you fuse an external LM for a new domain, that internal model is still there, pulling the search back towards the source domain and fighting the LM you added. \[5\] measures the gap directly: on cross-domain evaluation, shallow fusion gave a 16.1% relative WER reduction, subtracting the internal LM gave 29.1%.

The fix is to divide it out. `_fuse_lm` implements:

```
ln p_tot[k] = ln p[k] + λ·(ln(1 − p[∅]) + ln p_LM[k]) − λ_I·ln p_ILM[k]    for k ≠ ∅
ln p_tot[∅] = (1 + λ)·ln p[∅]
```

The `λ` half is eq. (3) of \[1\] unchanged (4.6); the `−λ_I` term is eq. (27) of \[5\]. Blank is untouched by the subtraction: a language model has no blank symbol, so `ln p_ILM[∅]` is defined as exactly `0`.

`lm_type` picks where `ln p_ILM` comes from.

**`"ilme"`** reads it off the transducer itself. Zero the encoder output and run the joint again; what survives is the label-history path alone:

```
z_ILM = J(g_u) = W_j·φ(W_p·h_pred + b_p) + b_j
```

which is eq. (25) of \[5\], and matches this repository's joint — `ffn_out(act(merge(ffn_enc(enc), ffn_pred(pred))))` — with `enc = 0`. Drop the blank logit, softmax over what is left, and that is `p_ILM`. This is exact and adds no parameters, but costs a **second joint call per decoding step**. `Transducer.call_next(..., return_internal_lm=True)` reuses the prediction network output it already computed, so the expensive recurrent half is not run twice.

**`"lodr"`** replaces the exact estimate with a cheap low-order n-gram trained on the same transcripts \[6\]. A bigram costs a table lookup instead of a joint call, and \[6\] reports it matching ILME in practice. `internal_lm` is an ordinary `LanguageModel` (4.6) with its own state, threaded through the beam and re-ordered onto the selected parents exactly like the external LM's. Its blank column is zeroed on the way in, so a general-purpose LM can be passed without special-casing.

Unlike the external LM, this one **does ship**: [`BigramLanguageModel`](../tensorflow_asr/models/lm/bigram_language_model.py), fitted from your training transcripts by

```bash
tensorflow_asr train_lm \
    --config-path=/path/to/config.yml.j2 \
    --datadir=/path/to/data \
    --dataset-type=slice \
    --target=internal \
    --output=/path/to/bigram.weights.h5
```

It counts adjacent token pairs over `data_config.train_dataset_config`, tokenizing with the config's own tokenizer so the indices line up with the transducer's vocabulary by construction. Note that a bigram is **not** trained by gradient descent: its maximum-likelihood estimate is a ratio of counts, exact in one pass, so it exposes `fit_counts` and `train_lm` dispatches to that instead of running `fit`. Any model without `fit_counts` — a neural LM — is trained by gradient descent on next-token cross-entropy instead. **The corpus matters**: the bigram must see the transcripts the transducer trained on, because that is what its internal LM learned. Counting the external LM's target-domain text instead would subtract the very knowledge fusion is adding, which is why the script takes no corpus argument.

Two details of the table:

- **Row `blank` is the sentence-start distribution.** The beam holds `last_token = blank` until a hypothesis emits its first label, so that row conditions the first token. There is no end-of-sentence counterpart — a hypothesis ends when the frames run out, not on an emitted symbol.
- **Smoothing is mandatory, not a refinement.** An unsmoothed table gives `ln 0 = -inf` for every pair that never occurred, and the beam adds that into hypotheses that are perfectly legal, killing them outright. `build_table` uses Jelinek-Mercer interpolation with an add-`delta` unigram, `p(w|v) = λ·p_bi(w|v) + (1−λ)·p_uni(w)`. `λ` is forced to 0 for a context with no counts so its row still sums to one, and `λ = 1` is **rejected** — it looks like a legal "trust the counts fully" setting but leaves a seen context with no unigram floor, sending its unseen successors to `-inf`.

The table is a dense `[V, V]` float32 **non-trainable weight**, saved and loaded through the ordinary keras h5 path like any other model: 256 KiB at `V = 256`, 4 MiB at `V = 1000`. The 20k-bigram pruning of \[6\] is not implemented — it exists for vocabularies far larger than this repository's, where dense storage stops being free.

Both are approximations of the same quantity, so use one or the other, not both.

**Starting weights.** Both papers tune `λ` and `λ_I` on held-out data, and the right values are dataset-specific. Their reported optima, as a starting point:

| Source                | Setup                                | external `λ`      | internal `λ_I`      | `λ_I / λ` |
| --------------------- | ------------------------------------ | ----------------- | ------------------- | --------- |
| ILME \[5\], RNN-T     | cross-domain, LibriSpeech test-clean | 0.30              | 0.14                | 0.47      |
| ILME \[5\], RNN-T     | cross-domain, LibriSpeech test-other | 0.24              | 0.12                | 0.50      |
| ILME \[5\], RNN-T     | intra-domain, dictation              | 0.26              | 0.20                | 0.77      |
| ILME \[5\], RNN-T     | intra-domain, meeting                | 0.08              | 0.03                | 0.375     |
| LODR \[6\], pruned RNN-T | in-domain / cross-domain          | 0.375 – 0.75      | 0.125 – 0.375       | ~0.2 – 0.6 |

So: `λ` in the low tenths, `λ_I` roughly **half of `λ`**, and both smaller when the model is already in-domain. LODR \[6\] uses a **bigram** pruned to the 20k most frequent, and notes that character-level units may need a higher order to have enough distinct n-grams.

One caveat specific to this implementation: `λ` here is ALSD++'s eq. (3) weight, which also scales blank by `(1 + λ)`. \[5\] tuned its `λ_T` against plain shallow fusion, which leaves blank alone. The numbers above are a starting region to sweep around, **not** drop-in values.

Practical notes:

- Tune `λ_I` **below** `λ`. Neither paper analyses the failure mode, but it follows from the formula: `ln p_ILM` is negative, so subtracting it *raises* label scores while blank is untouched — oversubtracting drives over-emission.
- `λ_I = 0` makes the correction an exact no-op, so it is safe to leave configured while sweeping.
- `lm=None` with `lm_type="ilme"` is legal and means "subtract the internal LM, fuse nothing". Unusual, but it is what \[8\]'s density-ratio framing degenerates to with a uniform external LM.
- `"shallow"` is the default, so a config written before any of this existed decodes bit-identically.

### 4.8 Usage

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

> **The TFLite export currently decodes without any language model.** `make_tflite_function` calls `recognize_beam(inputs, beam_width=beam_width)` and passes nothing else, so `lm`, `lm_alpha`, `lm_type`, `internal_lm` and `lm_beta` are all left at their defaults — even though `scripts/tflite.py` calls `model.make_lm()` first. This predates ILME/LODR and applies to plain shallow fusion just the same. Everything on this page about fusion holds for `model.recognize_beam(...)` and for evaluation via `predict_step`, but an exported `.tflite` is a plain ALSD++ beam.

The split follows what each thing *is*. The language models are models, so they sit in a top-level `lm_config` next to `model_config`; how to score with them is a decoding setting, so it sits in `decoder_config` next to `beam_width`:

```yaml
model_config:
  class_name: ...
  config:
    ...

lm_config:
  # the LM fused *in*. Optional.
  external_config:
    class_name: my_package>MyLanguageModel
    config:
      ...
  # the low-order LM subtracted. Required by lm_type: lodr.
  internal_config:
    class_name: tensorflow_asr.models.lm.bigram_language_model>BigramLanguageModel
    config:
      vocab_size: 1000
      blank: 0

decoder_config:
  beam_width: 16    # 0 (the shipped default) disables beam search
  norm_score: True  # -> score_norm
  lm_type: lodr     # shallow (default) | ilme | lodr
  lm_alpha: 0.3     # -> lm_alpha, eq. (3) lambda
  lm_beta: 0.15     # -> lm_beta, eq. (27) lambda_I. Keep it below lm_alpha.
```

Both model configs are ordinary keras serialization blobs, exactly like `model_config` — which is what they are, and nothing more. **Trained weights are not in the config**: they are passed to `tensorflow_asr test` as `--lm-h5` and `--internal-lm-h5`, beside the ASR model's own `--h5`. Same reasoning in all three cases — the config says what the model *is*, a checkpoint says which trained copy of it you happen to be running, and you sweep the latter without editing the former.

For `lm_type: ilme` there is nothing extra to configure — the estimate comes from the transducer's own joint — and `external_config` may be dropped entirely if you want subtraction without fusion. That combination warns, because far more often it means a config that forgot `external_config`.

Fitting the language models is a separate step, before evaluation, and `--target` names the key it builds:

```bash
tensorflow_asr train_lm ... --target=internal --output=.../bigram.weights.h5   # counting
tensorflow_asr train_lm ... --target=external --output=.../lm.weights.h5       # gradient descent
```

Those two outputs are what evaluation then loads:

```bash
tensorflow_asr test \
    --config-path=/path/to/config.yml.j2 \
    --h5=/path/to/asr.h5 \
    --lm-h5=/path/to/lm.weights.h5 \
    --internal-lm-h5=/path/to/bigram.weights.h5 \
    ...
```

Both LM flags are optional. Leaving one out builds that model with its **initial** weights rather than skipping it, which is never what you want outside a test — so if a configured LM seems to be doing nothing, check the flag before the maths.

`predict_step` calls `BaseModel.get_beam_decoding_kwargs()`, which returns nothing when `beam_width <= 0`. In that case the beam column of the evaluation output mirrors the greedy one rather than running a second decode — the same `beam_width > 0` convention `make_tflite_function` already uses.

`model.make_lm(config.lm_config, lm_weights=..., internal_lm_weights=...)` **builds and nothing else** — it makes whichever of the two the config describes, leaves the other `None`, and never looks at `lm_type`. It is a no-op when nothing is configured. `scripts/tflite.py` passes no weights, because the models it builds go unused either way — see the note above.

Judging whether the result is coherent is a separate step, `app_util.validate_lm(model, decoder_config, lm_h5=..., internal_lm_h5=...)`, which `scripts/test.py` calls straight after. Keeping them apart means building works on its own — in a test, a notebook, a half-configured sweep — without having to satisfy rules that only matter once you actually decode.

It **raises** for one case only: `lm_type: lodr` with no internal model, which cannot run. Everything else warns, because each is legal and occasionally deliberate, but far more often a lost config key or a lost flag — and every one of them fails *silently*, decoding happily to a quietly worse WER:

| Warning | What it means |
| ------- | -------------- |
| `beam_width` is 0 with an LM configured | beam search is off, so no LM is used at all |
| an LM is configured but its `--*-h5` is missing | it is fused in with **initial, untrained** weights |
| `lm_type` is not `shallow` and no `external_config` | the internal LM is subtracted with nothing fused in |
| `internal_config` built but `lm_type` is not `lodr` | it was built and will never be read |
| an `external_config` with `lm_alpha: 0` | a full LM call per step that changes nothing |
| `lm_type` not `shallow` with `lm_beta: 0` | the correction is computed and then multiplied away |

Unknown `lm_type` is caught earlier still, by `DecoderConfig` at config load, before any model is built.

Neither LM is tracked as a keras sub-layer of the ASR model, so their weights never enter its checkpoint — they have their own h5 files.

### 4.9 Deviations from the paper

- **Transcript storage.** \[1\] uses a trie (`transcripts` + `transcripts_ptrs` backlinks) to avoid copying whole transcripts on each expansion. Here transcripts are dense `[B, W, 2T+1]` and re-gathered each step, because `tf.gather` over the beam axis is a single vectorized op and keeps every shape static, which is what XLA and TFLite export need. Memory is `O(B * W * T)`.
- **No bundled n-gram external LM.** \[1\] evaluates against a GPU-resident 6-gram (their NGPU-LM, chosen so the LM call is cheap enough to rescore the full hypothesis set rather than prune early). What ships here is `LSTMLanguageModel`, matching \[5\]'s architecture instead — a dense `[V, V, ...]` n-gram table is only viable at order 2, which is why the *internal* LM is a bigram and the external one is neural. No pretrained weights ship for either; `train_lm` fits them.
- **Bigram only.** `BigramLanguageModel` is order 2, dense, unpruned. \[6\] notes character-level units may want a higher order; that needs a new class, since a dense `[V, V, V]` trigram is not viable and the state would have to carry the token before last.
- **Fusion form under ILME/LODR.** \[5\] states eq. (27) over plain shallow fusion, which leaves blank alone. Here the external term keeps ALSD++'s `(1 + λ)` blank scaling (4.6) and the subtraction is applied to the labels only. That combination is neither paper verbatim; it is the coherent merge of the two, and it keeps the deletion-rate property eq. (3) exists for.
- **CUDA graphs.** Not applicable — the loop is a `tf.while_loop`.

### 4.10 Limitations

- **Recombination is lossy when `s` binds.** Merging by transcript hash keeps the higher-scoring copy, which may be the one with *less* expansion budget left on the current frame. When `s` never binds this cannot happen and the search is exact.
- **Token budget.** `u` is capped at `max_frames * 2 + 1` to match the greedy batch decoder's output width. `max_frames` is the padded batch width, not the individual utterance length, so the effective cap for a short utterance in a mixed batch is looser than `2 * its own length + 1`.
- **`-inf` is `-1e9`**, kept finite so masked entries can be added to without producing `NaN`. Scores accumulate in `float32` regardless of the model's compute dtype.
- **LM state does not survive streaming.** `PredictOutput` carries `next_decoder_states` but has no field for LM state, so a fused LM restarts from `get_initial_state` on every chunk. This applies to LODR's low-order LM too. Both are therefore only correct for whole-utterance decoding. ILME is unaffected — its state *is* the prediction network state, which does survive.
- **`lm_alpha=0` with an `lm` set still costs a full LM call per step** — it is a no-op mathematically, not computationally. Pass `lm=None` to skip the work. The same holds for `lm_beta=0`: `"ilme"` still runs the second joint call, and `"lodr"` still runs the low-order LM. Set `lm_type="shallow"` to skip it.
- **ILME roughly doubles the joint cost.** The prediction network is reused, but the joint runs twice per step on all `B * W` rows. LODR exists precisely to avoid this.
- **No weight tuning is provided.** `λ` and `λ_I` are dataset-specific and both papers tune them on held-out data. There is no sweep helper in this repository.
- **TFLite export drops every LM setting.** See the note in 4.8. Pre-existing, and not specific to the internal LM correction.
- **No timestamps and no n-best output** — only the single best hypothesis is returned.

### 4.11 Verification

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
| ILME / LODR, eq. (27)          | corrected objective vs the same DP at `λ_I` in 0 / 0.4 / 0.5 / 1.0, with and without an external LM, both `score_norm` modes                                                 |
| `p_ILM` is a real distribution | blank column exactly `0`, labels sum to one without it, label ordering survives renormalisation                                                                              |
| Zeroing the encoder works      | on the **real** joint, not the stub: two very different acoustic frames give the same `p_ILM`, while the ordinary output differs                                             |
| `λ_I = 0` is a no-op           | both `"ilme"` and `"lodr"` at `λ_I = 0` decode identically to plain shallow fusion                                                                                          |
| LODR state threading           | the low-order LM is the same stateful test LM, so mis-ordering its state onto the parents fails                                                                              |
| Bad configuration is rejected  | unknown `lm_type` (at config load), and `"lodr"` with no `internal_lm` / no `internal_config`                                                                               |
| Config plumbing                | `get_beam_decoding_kwargs` for `beam_width <= 0`, `norm_score`, LM attachment, and that a config with no `type` yields exactly the pre-ILME argument set                     |
| Config plumbing, LM side       | `make_lm` builds only what is configured and ignores `lm_type` entirely, and routes each h5 path to its own model                                                            |
| Misconfiguration is caught     | `validate_lm` raises only for `"lodr"` with nothing to subtract, and warns on each of the six silent cases above; `DecoderConfig` rejects an unknown `lm_type` at load        |
| LM stays out of the checkpoint | attaching either LM does not change `model.weights`                                                                                                                          |

`tests/test_lstm_language_model.py` and `tests/test_train_lm.py` cover the shipped external LM and the training pipeline:

| Property                          | How                                                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| The two forward passes agree      | `call_next` stepped with threaded state reproduces `call` teacher-forced, over 1/2/3 layers x tied and untied                 |
| Outputs are log-probabilities     | both paths exponentiate to 1; state has the `[B, nlayers, 2, units]` layout the beam re-orders                                |
| The state carries the history     | the same token from two different states gives different distributions, so it is not a unigram in disguise                    |
| Usable where it is called         | runs inside a real `tf.while_loop`, graph output bit-identical to eager, survives config + h5 round trip                      |
| Matches the published size        | ILME defaults at V=3999 land within 55-61M parameters against the paper's 58M                                                 |
| Corpus reading                    | plain and gzipped text both stream; indices match the tokenizer; `max_lines` caps                                             |
| Teacher forcing lines up          | inputs are targets shifted by one with blank in front, padded to the longest in the batch, empty lines dropped                |
| Padding is masked from the loss   | weights are 1 on real tokens and 0 on padding, and the padded targets really do hold blank -- which is why it cannot be inferred |
| Training learns the corpus        | on a corpus with two possible openings, >90% of the mass after BOS lands on those two, and P(blank) stays under 0.05          |

`tests/test_bigram_language_model.py` covers the shipped LODR bigram separately:

| Property                         | How                                                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Counting is right                | sentence starts land in row `blank`, repeated pairs counted once each (not collapsed), out-of-vocabulary tokens dropped     |
| `fit_counts` fills the weight    | a fresh model is all zeros, a fitted one matches `build_table`, and the weight is non-trainable                             |
| h5 round trip                    | `save_weights` → `build_lm(blob, weights=path)` restores the table; the table is never inlined into the config             |
| Every row is a distribution      | labels sum to one, blank column exactly `0`, **no `-inf` anywhere**, at four interpolation weights                          |
| Unseen context backs off cleanly | a context with no counts still sums to one and equals the unigram exactly                                                   |
| The counts actually matter       | an observed successor outranks an unobserved one; `λ = 0` collapses every row to the same unigram                           |
| Unsafe smoothing is rejected     | `delta = 0` and `interpolation = 1` both raise — the second looks legal but sends unseen successors of a seen context to `-inf` |
| Same form as ILME                | table and `_internal_lm_log_probs` agree on blank-inert / labels-normalised, so `λ_I` means one thing across both           |
| The model works where it is used | `call_next` row lookup and untouched state, `call` scores a whole sequence and agrees with `call_next` step by step, runs inside a real `tf.while_loop`, int indices survive keras autocasting |

Beyond the suite, TFLite conversion of `recognize_beam` succeeds under `jit_compile=True`.

Two caveats on what this does **not** establish:

- There is no trained checkpoint in this repository, so these are correctness results against a known optimum, **not** WER measurements. In particular no claim is made that shallow fusion or internal LM subtraction improves accuracy on real audio — only that they compute eq. (3) and eq. (27) correctly. The WER numbers quoted in 4.7 are the papers', on their data.
- With a binding `s` and a narrow beam — the actual operating regime — exactness is not guaranteed and is not asserted; pruning and transcript-hash recombination are both lossy by design.

## 5. References

1. L. Grigoryan, V. Bataev, A. Andrusenko, H. Xu, V. Lavrukhin, B. Ginsburg. *Pushing the Limits of Beam Search Decoding for Transducer-based ASR models*. Interspeech 2025. <https://arxiv.org/abs/2506.00185>
2. G. Saon, Z. Tüske, K. Audhkhasi. *Alignment-Length Synchronous Decoding for RNN Transducer*. ICASSP 2020.
3. A. Graves. *Sequence Transduction with Recurrent Neural Networks*. 2012. <https://arxiv.org/abs/1211.3711>
4. V. Bataev, H. Xu, D. Galvez, V. Lavrukhin, B. Ginsburg. *Label-Looping: Highly Efficient Decoding for Transducers*. Interspeech 2024. <https://arxiv.org/abs/2406.06220>
5. Z. Meng, S. Parthasarathy, E. Sun, Y. Gaur, N. Kanda, L. Lu, X. Chen, R. Zhao, J. Li, Y. Gong. *Internal Language Model Estimation for Domain-Adaptive End-to-End Speech Recognition*. SLT 2021. <https://arxiv.org/abs/2011.01991>
6. Z. Yao, X. Yang, P. Żelasko, et al. *LODR: Low-order Density Ratio for Language Model Integration in End-to-End ASR*. 2022. <https://arxiv.org/abs/2203.16776>
7. A. Hannun, A. Maas, D. Jurafsky, A. Ng. *First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs*. 2014. <https://arxiv.org/abs/1408.2873>
8. E. McDermott, H. Sak, E. Variani. *A Density Ratio Approach to Language Model Fusion in End-to-End Automatic Speech Recognition*. ASRU 2019. <https://arxiv.org/abs/2002.11268>
