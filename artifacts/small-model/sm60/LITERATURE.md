# Literature review for the small-model factory (SM60), 2026-10-29

Question: which recent RWKV / linear-attention / state-space ideas could make the small model (SM72s, 331,511
parameters) more accurate, or the smaller model (SM73) smaller, without more card/note state bytes or much CPU time
per review. Papers 2024-2026, summarized in my own words. I read abstracts and, for Mamba-3, the full text; anything
seen only in secondary write-ups is flagged.

## What our block already is

Read from `rwkv/model/rwkv_model.py` (main, 10-29). Each layer is an RWKV-7 time-mix plus a ReLU-squared channel-mix
with token shift. Width 64 (4 heads x 16), 13 layer steps per review (card stream 2 layers, note stream 1, then the
deck / preset / user streams). Per layer the recurrent state is a 4 x 16 x 16 matrix (1,024 B at int8) plus two 64-d
shift vectors, about 1,520 B per layer; card + note = 4,554 B. Against the RWKV-7 paper it already has per-channel
decay, an in-context learning rate `a` in (0,1), the value residual from layer 0, a gate, the bonus term, and
per-head key/value scales. Initial states are zero. So several newer ideas (Gated DeltaNet's decay plus delta write,
Kimi Linear's channel-wise decay, a negative-eigenvalue range, a short input convolution) are already covered.

Test-cost yardstick (from the smaller-model design (SM73)): warm start from the epoch-20 checkpoint (SM72s) with new
parameters initialised so the function is unchanged; one 8k epoch about 106 min; then a quarter-epoch 32k cool-down
about 2 h, paired against the same cool-down on the unchanged checkpoint (464 set for the instant head, the 2,225 set
for the curve head). A from-scratch run is about 20 epochs, 1.5 days: avoid it.

## Ranked shortlist

### 1. Key-side micro-parameters: separate removal and insertion keys, plus learnable r/k biases

- **What.** RWKV-7 gives the delta rule two key roles: a normalized removal key (what to erase) and an insertion key
  (what to write), each with its own learned per-channel scale. We tie them (insertion = removal x `a`, with only a
  per-head scalar scale). Add per-channel learned vectors for both roles. Separately, Mamba-3 adds a learnable,
  per-head, per-channel bias to its B and C projections after normalization (our k and r are the analogues).
- **Papers.** RWKV-7 "Goose" with Expressive Dynamic State Evolution, 2025, https://arxiv.org/abs/2503.14456.
  Mamba-3: Improved Sequence Modeling using State Space Principles, 2026, https://arxiv.org/abs/2603.15569.
  Comba: Improving Bilinear RNNs with Closed-loop Control, 2025, https://arxiv.org/abs/2506.02475 (same idea of a
  correction key distinct from the write key).
- **Why it might help.** A constant key bias gives a head an always-present slot that does not depend on the input
  (a per-card or per-note baseline). In Mamba-3's 440M ablation the bias was the larger part of the gain (perplexity
  16.49 to 15.72), and it made an explicit short convolution unnecessary. Decoupled keys let the model erase without
  being forced to write along the same direction.
- **Cost.** About 4 x 64 values per layer, roughly 3.3k parameters over 13 layers (+1%). State bytes: none. CPU: a few
  vector operations per layer, negligible. The bias adds a constant write, so the int4 / int8 state ranges need a
  re-check.
- **Test plan.** Removal / insertion scales at 1 and biases at 0 (function unchanged), warm start from the small model (SM72s) epoch 20,
  one 8k epoch at constant LR, then the paired quarter-epoch 32k Muon cool-down against the unchanged checkpoint.
  About 4-5 GPU h with the reference arm. Run biases plus decoupled keys as one arm; split only if positive. Pass if
  the paired gain is at least 0.00015 curve or 0.00010 instant.

### 2. Learned initial state for the card and note streams

- **What.** New cards and notes start from an all-zero matrix and zero shift vectors. Make the starting state a learned
  parameter per layer (shared by all cards), as in initial-state tuning for state-space models.
- **Papers.** Parameter-Efficient Fine-Tuning of State Space Models (includes initial-state tuning), 2024,
  https://arxiv.org/abs/2410.09016; State-offset Tuning, 2025, https://arxiv.org/abs/2503.03499. Both show that the
  starting state alone is an expressive knob. I found no RWKV-specific primary work.
- **Why it might help.** The first review of a card is predicted from a blank card state plus the deck / preset / user
  streams. A learned prior (for example "a new card is fragile") takes a burden off the other streams. Expect a small
  gain, concentrated on first reviews.
- **Cost.** About 1,150 values per card/note layer, ~3.5k parameters. State bytes: zero (a never-reviewed card's state
  is recomputed from the weights). CPU: zero per review.
- **Test plan.** Same warm start, zero-initialised so it begins identical to the small model (SM72s); one 8k epoch plus paired
  cool-down, 4-5 GPU h. Read the gain split by review number (the existing gap-by-stage analysis (SM24)), since the effect
  should sit on the first reviews. Can share the epoch with candidate 1 as a two-arm run.

### 3. Recurrent-state quantization drift check, riding on the smaller model's stage 1 (SM73)

- **What.** A 2026 NeurIPS paper on quantizing the recurrent state of linear-attention and SSM models finds that a
  bit-width that is safe over short sequences can stop being safe as decoding runs longer, because quantizer scale and
  state feed back into each other; plain persistent group-wise int4 held up best. I saw only its listing summary
  (https://neurips.cc/virtual/2026/173810). Related: Quantizing Small-Scale State-Space Models for Edge AI, 2025,
  https://arxiv.org/abs/2506.12480 (post-training quantization hurts, quantization-aware training recovers it, mixed
  precision per component).
- **Why it matters here.** The smaller model is the int4 state (3,018 B). A phone stores the state between reviews, so
  a card with 100+ reviews is re-quantized 100+ times. The 732-set mean may hide drift on long-history cards.
- **Cost.** No training. Stratify the stage 1 (SM73) int4-vs-int8 paired delta by the card's review count (1-5, 6-20,
  21-50, 51+), ~10 min CPU on existing per-review outputs, if the scorer re-quantizes after every review (check this
  first; otherwise add that mode). Concern if the 51+ bucket is more than twice the mean loss. The fix is cheap:
  higher-bit shift vectors or a refit per-row scale (see the smaller int state, round 2 (SM49)).

### 4. Two-step delta update (DeltaProduct, n = 2) in the card stream only

- **What.** Two delta-rule steps per review instead of one, so the transition is diagonal plus rank 2. DeltaProduct:
  Improving State-Tracking in Linear RNNs via Householder Products, 2025, https://arxiv.org/abs/2502.10297. It beats
  the one-step delta rule on state tracking and language modelling at unchanged state size.
- **Why it might help.** A review carries several signals (rating, elapsed time, duration); two writes let the card
  state record "was hard" and "interval was long" as separate directions.
- **Cost.** Extra key, value and `a` paths per card layer: about 10k parameters per layer, ~+20k (+6%) for the two card
  layers. State bytes: unchanged. CPU: card time-mix up 30-40%, whole step maybe +8%. The most expensive candidate and
  the only one touching the CUDA kernel (single-step WKV today).
- **Test plan.** Warm start with the second step's write weights at zero (a no-op), one 8k epoch plus paired
  cool-down, ~5 GPU h plus kernel work. Do it only after 1 and 2 and only if one lands: the evidence is synthetic
  state tracking and 1B-scale language models, and card histories are short.

### 5. Own idea, weakly literature-backed: drop the channel-mix token shift in card and note layers

- **What.** Each layer stores two 64-d shift vectors (about 250 B of its ~1,520 B). Mamba-3 found its explicit short
  convolution redundant once the bias was in (adding it back worsened perplexity, 15.72 to 15.85, at 440M). If
  candidate 1 lands, the channel-mix shift in the three card/note layers may be droppable: roughly 8% of card + note
  state (~370 B of 4,554).
- **Test plan.** After candidate 1, fine-tune with `lerp_k` forced to zero in those layers and compare in a paired
  cool-down, ~5 GPU h. Only the card/note layers matter for phone RAM.

## Looked at, skipped, why

- **Gated DeltaNet** (https://arxiv.org/abs/2412.06464) and **Kimi Linear / Kimi Delta Attention**
  (https://arxiv.org/abs/2510.26692): decay plus delta write. RWKV-7 already has per-channel decay and a vector
  learning rate, so it is a superset.
- **Mamba-3 exponential-trapezoidal step**: an implicit width-2 input convolution; token shift already is one.
- **Mamba-3 complex (rotary) state** and **negative eigenvalues** (https://arxiv.org/abs/2411.12537): gains shown on
  parity-style synthetic tasks. Our `a` with a normalized key already reaches eigenvalues near -1; review sequences are
  short and not parity-like.
- **Mamba-3 MIMO**: up to 4x decode FLOPs at the same state, the wrong direction for phone CPU.
- **Titans** (https://arxiv.org/abs/2501.00663): deep memory with a gradient step per token; more state, far more CPU.
- **mHC / hyper-connections** (https://arxiv.org/abs/2512.24880): wider residual stream, reported at large scale only,
  no state saving, extra mixing per layer.
- **RWKV-Lite / RWKV-edge** (https://arxiv.org/abs/2412.10856): compresses weights. Ours are 0.44 MB int8, under 1% of
  the state at 14k cards.
- **RWKV-8 ROSA / DeepEmbed**: only third-party descriptions found, nothing primary; ROSA targets discrete token
  copying, our inputs are review features. Revisit if a paper appears.
- **Irregular time-series RNNs** (e.g. https://arxiv.org/abs/2510.16161): elapsed-time inputs are already in the
  pipeline; nothing new for the state.
- **Weight tying / shared layers**: cuts parameters only, which do not limit the phone (SM73).
- **Search gap**: no 2026 RWKV-7 successor paper turned up; the freshest relevant work is Mamba-3 and the
  state-quantization paper.

## Suggested order

Candidates 1 and 2 as a two-arm run sharing one warm-start epoch (about 8 GPU h in all, with the reference
cool-down), candidate 3 now on CPU, 5 only after 1, 4 only if 1 or 2 shows the card stream gains from richer writes.
A gain from 1 or 2 also counts as the buy-back lever the smaller model's stage 2 (SM73) is waiting on.
