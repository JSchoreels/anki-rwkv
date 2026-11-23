# Opportunities from how outside research found its state-compression gains (11-23)

A thinking and writing task: no jobs were run. One synthetic CPU sanity check (under a minute, section 3, idea 1) was run on random matrices; no project data was read for it. Sources: two outside research bundles from a state-quantization research arm on the same dataset family (their rank-2 bundle (row 17) and their tail-penalty bundle (iter 55)), of which only the Markdown files were read (README, ANSWERS, verification, MANIFEST, and the three design notes shipped in their code folder: the deploy-functions note, the experiment protocol and the engine port and batching plans). No code, binary or script from the bundles was opened or run, and no code is copied here; ideas only. Our side: `AGENT.md`, `docs/GPU_OPTIMIZATIONS.md`, `pages/data/factory-curated.json` (194 result rows, 185 plan rows, 73 backlog rows) and the RESULT files of the state-compression line and its neighbours.

Rules used below: the accuracy go rule is a 95% CI below 0 and an effect of at least 0.05%; the state-format budgets are Big at most +0.15% against its own int8, Small at most +0.5% and Tiny at most +1.0% in total against the matching Big model at int8 (CI upper bound per head). Already running and not proposed again: the rank-2-then-4 format (T24) in the fork and smaller shift vectors and factors on T24 states (SM152). Already rejected and not proposed again: rank 1 as a uniform format, the tail-energy penalty and rank-2 QAT on our models.

## 1. How they worked: the method behind their findings

Each pattern names where it is described in their bundles and the math tool behind it.

**P1. Isolate the factor with a diagnostic arm before optimising it.** Their "cn8" arm stored the full matrix at int8 with no rank truncation and found it nearly free (+0.000033 / +0.000022 LogLoss), which proved that the whole tax of their rank-1 family came from the rank truncation and none from precision; that one arm decided that bits should go to rank (rank 2 on the card) rather than to finer codes, and why 4-bit factors fail (tail-penalty bundle (iter 55) README section 6, "Why rank, not precision"). Tool: an ablation that holds every factor but one fixed, so a sum of two effects becomes two measured terms.

**P2. Turn storage into a bit-accounting function and set bars relative to a naive baseline.** Every format is priced by one function (`bits_for`: layers times (2 times rank times K times factor bits plus 64 times shift bits)), so a card at rank 2 int8 with int4 shifts is 1,280 bits and a note at rank 1 is 2,304; the gate bar is half of the naive all-int4 state (8,704 bits), and the report states sizes as a share of fp32 (2.6%) and of naive int4 (4.9 times smaller) (the tail-penalty bundle (iter 55), README section 1). Tool: an explicit cost model, which turns format design into a constrained optimisation instead of a list of labels.

**P3. Per-stream scopes.** Rank and precision are set per stream with one string (`card:2:int8,note:1:int8`, shifts per stream), the card gets rank 2 and the note rank 1, and the deck, preset and user streams stay fp32 because they are few (the tail-penalty bundle (iter 55), README section 1 and ANSWERS section g). Tool: treating each stream as its own source in a rate allocation.

**P4. A precision ladder measured once and reused.** At rank 1 or 2, int8 factors are essentially free, int6 costs a little, int5 and int4 are the knee, int3 and int2 fail (ANSWERS section g). Tool: a monotone distortion curve per component, which tells where the next bit is worth most (the quantization-theory view: error power falls about 6 dB per bit until the signal's own structure dominates).

**P5. Spectral view of the state and an argument for why truncation does not drift.** They measure the "tail energy" (the share of squared singular values beyond rank r) per head, and argue from three measurements that re-truncating the current state at every write does not accumulate like per-entry rounding: the oldest cards are where the truncated state does best, the longest users are not the damaged ones, and a 29,166-review history scores better with the quantizer than without (ANSWERS section c). Tool: SVD and the Eckart-Young theorem (best rank-r approximation in Frobenius norm), and the observation that a projection re-estimated from the present state forgets old errors instead of integrating them.

**P6. A shaping penalty sized from a measured pre-run bracket.** The tail-energy penalty weight was fixed at 1.0 because the pre-run tail was about 0.0245, so the term was about 2.4% of the loss at the start; the penalty is accumulated on one write step in eight per stream tag to keep the backward graph within memory, and it needs the unfused per-step loop because the fused kernel truncates internally (the tail-penalty bundle (iter 55), README section 2). Tool: a regulariser on singular values (a nuclear-norm-like term restricted to the tail), with its weight set in units of the loss.

**P7. Two pairings, never mixed.** The acceptance gate compares the trained-with-format model against a control that had the same short decay without the format (so it mixes the training gain and the format tax), while the deploy tax compares the same weights with the quantizer on and off; the two differed by about ten times and are reported separately (the tail-penalty bundle (iter 55), README section 3.1a). A caveat their own answers spell out: their ceiling of +0.0005 absolute is about 0.15% to 0.16% relative, about three times looser than a +0.05% bar (ANSWERS section e).

**P8. Answer-count buckets and a per-card movement guard.** Every format is read by the card's answer-count bucket (1-2 up to 128+) and by the per-card maximum change in predicted recall (share of cards with a review moving by more than 0.05 or 0.10) (ANSWERS sections a and b). Tool: stratified paired differences and a max-over-card statistic, which catch damage that a dataset-wide mean hides.

**P9. Fixed bases fitted offline.** Their smallest passing rows ship a fitted basis with the model and store only an m by m coefficient block per note write (m = 4 or 6): 2,432 to 2,912 bits in all, and 7.1 microseconds per write at m = 4 against 41 to 49 microseconds for a rank-2 solve (ANSWERS sections f, g and 7). Tool: a fixed two-sided projection (the Tucker-2 or higher-order SVD idea: a basis per side learned from many states, a small core per entity).

**P10. Fast solvers with a reference twin.** The reference rank-r solve is a Gram matrix plus a symmetric eigendecomposition (about 10.5 microseconds for rank 1, 41 to 49 for rank 2 per layer per write); a fast path uses block subspace iteration warm-started from the previous factors (one or two iterations) and is kept out of parity runs because it is approximate (ANSWERS section f). Tool: Gram matrices, symmetric eigensolvers and warm-started subspace (power) iteration.

**P11. A trace-driven deploy engine and three-way parity.** The deploy engine runs on exported traces with no training data, reproduces the Python recurrent model to 5.9e-07 per review, and has its own finiteness observer that can exit on one non-finite value (verification sections 2 to 4). Their parity discipline caught a live cross-wiring bug that every shape check passed (deck and note states swapped under a reordered stream list) and two deploy feature columns that had drifted from the training rows, which a trace checked only against the Python path that built it could not see; the reusable rule is "a deploy feature builder is right if and only if its vector equals the training row for the same review" (engine port plan, gaps 7 and 8; deploy-functions note, section 4).

**P12. Deploy speed anchored on named deploy functions and measured with paired simultaneous trials.** Speed is the throughput of the function that builds the five stream states for one review, measured as simultaneous before and after runs over the same frozen batches with a one-sided Wilcoxon test over 20 trials; a 4.5 times arithmetic cut bought only 1.24 times wall-clock, and the batched query path has a cache knee at a batch of 128 with thread count irrelevant (protocol note; deploy-functions note; batching plan).

**P13. De-bundle controls.** When a shipped change bundled a stream reorder and an interleaved layer schedule, a control measured each part and found the schedule worth +0.000489 / +0.000612 while the reorder alone was slightly negative (engine port plan, gaps 7 and 8). Tool: a factorial split of a bundle.

**P14. Score what ships.** Their grade-order rectifier was moved from the loss into the model, so evaluation and the engine score the rectified model; the interval solver applies it at every bisection probe (engine port plan, gap 6).

**P15. Deploy contracts for every input, written as exact online rules.** Show-time clock shifts, a duration imputation for query rows with a shrinkage weight w = n_card / (n_card + k) between the card's and the user's running medians, and per-collection counters, each with its own parity check against the training rows (deploy-functions note, sections 4b, 4c, 5 and 6). Tool: empirical-Bayes shrinkage for a quantity that does not exist yet at prediction time.

**P16. A verification bundle per claim.** Weights, traces, evidence files, a manifest with SHA-256, a script that reproduces the checks from a fresh extract, and an explicit "what the numbers do not say" section (the tail-penalty bundle (iter 55), README section 3.3; verification section 8).

**P17. Tools visible only by file name.** Their manifest lists modules for sharpness-aware minimisation, adaptive weight decay, Muon, a "self feedback" loop, an FSRS stream, gradient statistics and a deck tree. We did not read them and do not know what they found with them; they are pointers only.

What did not transfer, and the likely reasons: their tail-energy penalty and their short format-aware decay ran 124 steps on 100 users of a 192,800-parameter model with one card layer and three note layers, and were judged against the looser absolute ceiling and against a no-format control trained the same way. On our Small two-head (SM118), the rank-2 cool-down (SM149) recovered only a third of the rank-2 cost over 2,474 update passes, and the tail-energy penalty cool-down (SM149b) cut the tail energy five times (their run: about 45 times) but made the model itself about +0.20% worse on the instant head at int8.

## 2. Where we already used each pattern, and where we did not

| Pattern | Where we used it | What came out | What is still open |
|---|---|---|---|
| P1 isolate the factor | The low-rank structure probe (SM147) checked the int8 reference against the float state (within 0.006%), and the low-rank int8 states test (SM148) reports the size cost and the format cost separately for Small and Tiny | Our rank cost is also a rank cost, not a precision cost | Not used for the Small-against-Big accuracy gap beyond the head-only fine-tunes (SM50) and the context-length check (SM13) |
| P2 bit accounting, relative bars | Bytes per card and note in every state test since the int state formats (SM47, SM49); percent against int8 in the low-rank tests (SM148, SM150); the user's 11-23 budgets | Clear byte tables per format | No single rate-distortion frontier across all formats tested (bytes against CI upper cost, with the convex hull) |
| P3 per-stream scopes | Card-only variants in the second integer-state experiment (SM49) and the six-layer Tiny with the card-only state (SM134); notes follow cards in the earlier int8 switch (SM144) | Notes and cards mostly got the same rank in the low-rank tests (SM148, SM150) | Rank per stream, per layer and per head, although the low-rank structure probe (SM147) shows large spreads (section 3, idea 3) |
| P4 precision ladder | Tiered card state by answer count (SM141), earlier int8 switch (SM144), int3 young tier (SM145), aggressive state tiers for Tiny (SM146), smaller shift vectors and factors on T24 states (SM152, running) | int8 rows robust, int4 drifts on long cards (SM136), int3 young tiers pass for Tiny | Precision per singular component (the first pair needs more bits than the second) |
| P5 spectral view | The low-rank structure probe (SM147): energy by rank, bucket, layer and head | Rank 4 keeps 97% to 99.5%, rank 2 87% to 95%, notes less low-rank with age | Weighting the truncation by what the model reads (section 3, idea 2) |
| P6 shaping penalty | The tail-energy penalty cool-down (SM149b) | Rejected: the model got worse | Nothing proposed |
| P7 two pairings | "Against its own int8" and "against the original" in the rank-2 cool-down (SM149) | Kept apart in our reports | Done |
| P8 buckets and per-card guard | R3 long-card rule and the G guard in the tiered and low-rank tests (SM141 to SM150) | Became our standard read-out | Long-card checks are underpowered and lose the longest users to memory (section 3, idea 6) |
| P9 fixed bases | The training-free state check (SM116, per-block PCA, 70% of values at +0.015% / +0.032%); codebook state compression (SM32, SM44, SM46, SM90); learned codebooks (SM32c) | Large codebooks cost 12 MB to 30 MB and 3.6 ms per review; the PCA projection was nearly free at 70% | A small two-sided basis (a few KB) with an m by m core has not been tried (section 3, idea 5) |
| P10 fast solvers | The GPU low-rank path in the low-rank tests (SM148, SM150) is a torch step loop with a Jacobi solve | GPU scoring at 0.28 to 0.46 times int8 speed, CPU at 0.86 to 0.96 times | An exact factored update makes the solve tiny (section 3, idea 1) |
| P11 trace engine, three-way parity | Native against PyTorch checks in the fork (Tiny in the fork (SM112), Big pair in the fork (SM109)), discrete-codec tolerance rule | Engine-level parity is routine | A check that the fork's own feature builder reproduces the training rows for the same review is not in the record we read (section 3, idea 12) |
| P12 deploy speed | The fork speed test (SM111, SM111b) | State build 5.6 s to 23.3 s per profile | The query-batch size in the fork was never swept against the cache knee (section 3, idea 13) |
| P13 de-bundle | The data-order re-seed (SM80), the control decay (SM42) | Removed false credit twice | Fine as is |
| P14 score what ships | The grade order by construction (SM95), grade-order sanity (SM33), rolling recalibration (SM125, SM128, SM140) | In place | The two instant predictions computed side by side in the "both models" fork mode are never combined (section 3, idea 9) |
| P15 input contracts | Show-time inputs since the show-time retrain (SM92); duration-leak scoring (SM31) | Show-time is the standard | Query-row duration is a constant; the shrinkage imputation was not tried (section 3, idea 10) |
| P16 verification bundle | Release packages with SHA256SUMS and test vectors (the RWKV family release (SM108)) | Done for releases | Not for experiment results; low priority |
| P17 named tools | Muon (the Muon best model (SM36)); checkpoint averaging (SM51); epoch weight averaging (SM74) | Muon kept; averaging no free gain | Sharpness-aware minimisation not tried (section 3, idea 11) |

## 3. New opportunities

Expected sizes are my estimates unless a measurement is cited; they are not results.

### Idea 1. Exact factored rank-(r+1) update for low-rank states (no GPU for the proof)

- Pattern: P10 (cheap solvers), made exact instead of approximate.
- Observation: our recurrence writes `S' = S (diag(w) - kd (a kd)^T) + v k^T` per head (the reference forward of the released small model). If the stored S has rank r as U V^T, then `S' = [U | v] [diag(w) V - (a kd)(kd^T V) | k]^T` has rank at most r + 1 in factored form. The best rank-r approximation of S' is therefore two thin QR factorisations of K by (r + 1) matrices and one SVD of an (r + 1) by (r + 1) core, about K (r + 1)^2 operations instead of a dense K by K solve (about K^3). The read `S' r` costs O(K r) from the factors instead of O(K^2) from the matrix.
- Sanity check run (synthetic random states, 300 writes, K = 16 and 32, r = 2 and 4): the factored path equals the dense SVD truncation to a maximum relative difference of 2.4e-14 to 3.3e-14. NumPy per-write timings are dominated by call overhead and say nothing about real speed; the operation count is 144 to 800 against about 4,096 to 32,768.
- Hypothesis: the same predictions as the rank-2 (Q2) and rank-2-then-4 (T24) formats of the low-rank int8 states test (SM148), within the discrete-codec tolerance rule (int8 rounding of the factors can flip on float noise), at a speed at or above int8 rows on the CPU, and without the 0.28 to 0.46 times GPU scoring slowdown of the Jacobi step loop. The switch from rank 2 to rank 4 at 16 answers needs no special case: the rank grows by at most one per write.
- Cheapest decisive test: a NumPy implementation behind the existing CPU scorer's low-rank label, compared on the 3 smoke users against the rank-2 (Q2) and rank-2-then-4 (T24) rows of the low-rank int8 states test (SM148) under the discrete-codec rule, plus a per-write micro-benchmark in a compiled language on the fork side. CPU only, about half a day of agent work; then hand it to rank-2 states in the fork (SM151) as the fork's update path.
- Expected win: every future low-rank GPU scoring pass about 2 to 3 times faster (it is now 105 minutes for the two Big models in the rank-1 int8 states test (SM150)); in the fork, low-rank states that are cheaper to update and read than int8 rows, turning the speed warning of the low-rank tests into a speed gain.
- Risks: near-tied singular values can swap order (harmless after truncation unless r cuts between them, which the tolerance rule covers); float32 QR on near-zero new columns (the first write) needs a guard; it must match whatever factor scaling convention the fork adopts.
- Rules: an exact change judged by parity, not by the accuracy rule; speed claims need a `--timing` run in a quiet window.

### Idea 2. Output-aware truncation: weight the rank-r fit by what the model reads (CPU first)

- Pattern: P5 plus P9 (a fixed matrix fitted offline), with the activation-aware low-rank idea used in model compression.
- Observation: the state is only ever used through products on its key side: the read `S r`, the removal term `S kd` and the decay on the key axis. Frobenius-optimal truncation treats every key direction as equally important. The weighted Eckart-Young problem, minimise the expected read error `E ||(S - S_hat) r||^2 = ||(S - S_hat) M^(1/2)||_F^2` with `M = E[r r^T]`, is solved exactly by truncating `S M^(1/2)` and multiplying back by `M^(-1/2)`; a variant adds `beta E[kd kd^T]` for the removal term; a third uses a discounted observability Gramian (the balanced-truncation tool from control theory), `M = R + D M D` with D the mean per-channel decay of the head, solved once as a small Stein equation, so directions that decay fast count less.
- Hypothesis: at the same bytes as rank 2 (Q2), the rank-2 cost of the low-rank int8 states test (SM148) (Big two-head (SM91s2) +0.080% / +0.109%, Small two-head (SM118) +0.04% / +0.08% against their own int8) falls by a third or more, which widens Big's thin margin (Big Instant-Only (SM99) at +0.132% against the +0.15% budget) for the format that rank-2 states in the fork (SM151) is shipping. M is one K by K matrix per layer and head (Small: under 4 KB at float16), shipped with the model; with idea 1 the factored update still applies, since `S' M^(1/2)` is still rank r + 1.
- Cheapest decisive test: fit M per layer and head from receptance (and key) statistics on a few hundred training users, then score the 8 users of the low-rank structure probe (SM147) on the CPU at rank 2 plain and rank 2 weighted (three variants), about an hour of CPU; if the rank-2 cost falls by at least a third on the six models, score the six release models on the 732 set on the GPU at rank 2 and rank 2 then 4 weighted (about 3.5 GPU hours, as the rank-1 int8 states test (SM150) took).
- Expected win: a third to a half of the rank-2 cost (a guess; it depends on how anisotropic the receptance statistics are), or the same cost at fewer bytes through idea 3.
- Risks: if receptance vectors are close to isotropic the gain is small; the out group norm makes the output scale-invariant per head, so a direction-weighted metric may be better than a norm-weighted one; statistics must come from training users only.
- Rules: a state format, judged by the 11-23 budgets; fits CPU-alongside-GPU for the pointer.

### Idea 3. Rate-distortion allocation of rank and factor precision per stream, layer and head (CPU first)

- Pattern: P2, P3 and P4 combined into one optimisation; tool: reverse water-filling and greedy marginal allocation.
- Observation: the low-rank structure probe (SM147) found a wide spread inside one model: Big card layer 0 keeps 96% to 98.7% at rank 2 while Big card layer 2 head 1 keeps 84% at rank 2 and 96% at rank 4; notes lose low-rank structure with age while cards stay flat. A uniform rank over every head spends bytes where they buy nothing and starves the weakest heads.
- Hypothesis: at the bytes of rank 2 (Q2), a per-head allocation (for example rank 3 or 4 on the weakest heads, rank 2 elsewhere, and rank 2 then 3 on notes with many answers) cuts the dataset-wide cost by 20% to 40%; or at the cost of rank 2 it saves another 10% to 25% of bytes. Precision per singular component follows the same logic: the second singular pair carries a quarter or less of the first one's energy, so it can take fewer bits than the first (this is per component, unlike the uniform int6 factors of smaller shift vectors and factors on T24 states (SM152)).
- Cheapest decisive test: step 0 is free, a single rate-distortion table from existing rows (the tiered, low-rank and compact-state tests (SM131, SM141, SM144 to SM150), bytes against CI upper cost per model, with the convex hull and the cost per saved byte), which also shows which labels are dominated. Step 1 on the CPU: per-head sensitivity on the 8 probe users of the low-rank structure probe (SM147) by truncating one head group at a time, cheaper with idea 1; allocate greedily by marginal LogLoss per byte. Step 2 on the GPU: the two best allocations on the 732 set for the six release models (about 3.5 GPU hours).
- Expected win: medium on bytes, small on accuracy; it compounds with idea 2.
- Risks: per-head sensitivities measured on 8 users are noisy; a heterogeneous layout adds bookkeeping in the fork; if the allocation wants rank 1 on some heads, that needs the user's explicit OK, because rank 1 was rejected as a uniform format.
- Rules: the 11-23 budgets; step 0 and step 1 run alongside GPU jobs.

### Idea 4. Energy-triggered rank growth per entity

- Pattern: P5 (tail energy measured at each write) used as a deploy-time decision instead of a training penalty, in place of the age trigger of the rank-2-then-4 format (T24).
- Hypothesis: with idea 1 the (r + 1)-th singular value of each write is available for free, so an entity can stay at rank 2 and grow by one rank (up to 4) only when the discarded share exceeds a threshold. This puts bytes on the notes with many answers that the low-rank structure probe (SM147) found least low-rank, rather than on every entity past 16 answers; at the mean bytes of the rank-2-then-4 format (T24) it should cost less, or at its cost use fewer bytes.
- Cheapest decisive test: CPU pointer on the 8 probe users with three thresholds (about an hour); GPU 732 set for the best threshold against the rank-2-then-4 format (T24) at matched mean bytes (about 3.5 GPU hours).
- Expected win: small to medium; the main value is a format that adapts to the user's actual card histories.
- Risks: variable-size records in the fork (one rank byte per entity, and storage that tolerates growth); a rank that never shrinks; threshold drift between GPU and CPU on near ties.
- Rules: the 11-23 budgets; wait for the outcome of smaller shift vectors and factors on T24 states (SM152) so the comparison uses its shift format.

### Idea 5. A small fixed two-sided basis with an m by m core (Tucker-2), mainly for Big

- Pattern: P9.
- Hypothesis: one value-side and one key-side basis per layer and head (top eigenvectors of the summed Gram matrices `sum S S^T` and `sum S^T S` over training entities, a few KB in all) and a stored m by m core per entity: Big at m = 4 stores 16 codes per head against 128 for rank 2 (a Big card at about 1,050 bytes against 2,419 at rank 2 and 13,891 at int8 rows, counting the unchanged 835 bytes of shift vectors and other values), at the cost of two small products per write and no solver. For Small (K = 16) the saving over rank 2 only appears at m = 4 (16 codes against 64), which is likely too lossy (see the risks).
- Cheapest decisive test: CPU only, energy captured by the fixed subspace for m = 4, 6 and 8 on the 8 probe users (state dumps from the CPU scorer, as the training-free state check (SM116) did for its PCA), compared with per-entity rank 2; continue to a scoring pass only if m = 6 keeps at least the energy of rank 2. A hybrid (fixed basis plus a per-entity rank-1 residual) is the fallback.
- Expected win: large in bytes for Big if it holds, nothing for Small and Tiny.
- Risks: the training-free state check (SM116) needed 4 to 74 of 256 directions per head for 90% of the variance, a wide spread, so a fixed m = 4 basis may miss entity-specific directions; the outside research's note codec cost about +0.1% on its immediate head, over our Big budget once the size of their model is taken into account.
- Rules: the 11-23 Big budget; CPU pointer only until it shows energy parity.

### Idea 6. A recurrent-mode long-card scorer with memory bounded by the state (no GPU)

- Pattern: P11 (a step engine that scores the longest histories without truncation, in minutes).
- Observation: our CPU scorer's peak memory grows with the review count (about 3.2 GB at 102,545 reviews and 5.6 GB at 150,500, the long-card state drift check (SM136)), so the memory guard removed the 15 to 16 longest users per size in that check, 13 long-card users for Big in the low-rank int8 states test (SM148), and the trained smaller state for Tiny (SM137) failed its long-card check on a wide CI (excess -0.008% [-0.175, +0.412]) with only 50 users.
- Hypothesis: a scorer that streams reviews one step at a time and keeps only per-entity states (the released model's reference recurrence already has this shape) gives the same per-review predictions as the current scorer within the strict 1e-5 rule, at memory set by the number of entities, so every long-card user fits and a larger pre-registered long-card cohort (for example 150 to 200 held-out users with 128+ answer cards from the 1,500 validation users) becomes affordable.
- Cheapest decisive test: exactness against the current CPU scorer on the 3 smoke users and 5 long-card users, then the missing Big users of the low-rank int8 states test (SM148) at rank 2 (Q2), where Big's margin is thinnest. CPU only, about a day of agent work.
- Expected win: no accuracy by itself, but it removes the coverage shortfall that keeps several decisions open, and gives long-card rules enough power to say "no loss" rather than "cannot tell".
- Risks: speed of a pure step loop on the CPU (acceptable for a few hundred users); the user-split rules of the eval pools must hold for any new cohort.
- Rules: measurement infrastructure, benchmark users only; it decides between existing formats rather than producing a model, so run it alongside GPU work.

### Idea 7. Shrink the answer-time head's between-card spread (CPU only)

- Pattern: P7 and P14 (calibrate what ships against a naive baseline), using a structural measurement first.
- Observation: answer-time expressiveness (SM142) found that the time heads separate fast from slow cards within one user and grade clearly better than the card's own running median, but spread the cards about 1.7 times too widely (between-card slope 0.59).
- Hypothesis: an empirical-Bayes shrinkage of each card's predicted time distribution toward the user's running per-grade distribution, `p' = alpha p_card + (1 - alpha) p_user` or a location shift by the factor 0.59, with alpha fitted offline per model (or causally per user, like the rolling recalibration of the timed cells), lowers time LogLoss while leaving within-user AUC unchanged (a monotone shrink keeps the ranking within one user and grade).
- Cheapest decisive test: the saved full time distributions of answer-time expressiveness (SM142), CPU only, under an hour: time LogLoss with a paired user bootstrap, then the 21 timed cells after the existing rolling recalibration (SM125, SM128, SM140), since Platt scaling may already absorb part of it.
- Expected win: time LogLoss -0.5% to -2% (a guess), better-calibrated per-card answer-time estimates in Card Info.
- Risks: the rolling recalibration may already fix most of the over-dispersion for the timed cells; the gain is on the time head, not on recall.
- Rules: the go rule on time LogLoss (CI below 0 and at least 0.05%); runs alongside anything.

### Idea 8. Session and rolling-load inputs for the answer-time head only (check first, then a head-only retrain)

- Pattern: P15 (inputs defined as exact online rules: reviews already done in the sitting, reviews in the last 10 and 60 minutes, and similar counters).
- Observation: the session position effect (SM85) found the same card answered 27.6% faster at reviews 201 to 400 of the day and 34% faster from the 401st, in 95% of users with 5,000 or more reviews; feature discovery (SM70) found no input that helps the grade, but it did not target the answer time.
- Hypothesis: the answer-time head, trained on a frozen trunk as in the answer-time head on richer frozen features (SM82c), gains clearly from two or three session counters, because the speed-up is real for the time and invisible to the grade. The grade outputs stay bitwise unchanged.
- Cheapest decisive test: first check whether the simple time features of the answer-time head on richer frozen features (SM82c) already include session position (if yes, drop this idea); otherwise one head-only retrain on one model (about an hour of GPU, by analogy with the time heads for the family (SM117)) scored against the current head.
- Expected win: time LogLoss -1% to -3% (a guess), relevant to the answer-time display and to any time-aware scheduling.
- Risks: the late-session speed-up may partly be rushing (the reason ordering by predicted time (SM89) was dropped), which a time head should still predict but a scheduler should not reward.
- Rules: the time-head gate (grade bitwise unchanged, beats both baselines) plus the go rule on time LogLoss.

### Idea 9. A free deploy-time ensemble of the two instant predictions (CPU first)

- Pattern: P12 and P14 (anchor on what the deploy path already computes; the overlap the outside research notes between its two deploy functions).
- Observation: in the fork's "both models" mode the two-head model and the Instant-Only model of a size both run on every review, and the two-head model's own instant head is computed and thrown away. Ensembles were measured only as distillation teachers (big model pairs, 0.03% to 0.09%, in the better teacher (SM65) design), never as a shipped predictor.
- Hypothesis: a fixed logit average of the two instant predictions, with one weight fitted on training users, beats the Instant-Only model alone by at least 0.05% at zero extra deploy cost.
- Cheapest decisive test: per-review instant predictions of both models of one size on the 732 set (one dump pass per model on the GPU, about an hour, or CPU on a subset of users), then a CPU paired bootstrap.
- Expected win: 0.02% to 0.08% on instant (a guess); the two models share a lineage (the Instant-Only models start from two-head checkpoints), so their errors are correlated and the gain may stay under the 0.05% line.
- Risks: small gain; needs both models loaded, which is already the case in that mode only.
- Rules: the accuracy go rule.

### Idea 10. Duration imputation on show-time query rows (CPU screen first)

- Pattern: P15 (the shrinkage rule `w = n_card / (n_card + k)` between the card's and the user's running median of past durations, the same value on the query row and on the four what-if rows).
- Hypothesis: our query rows carry a constant "no press yet" duration; a model trained with the imputed value predicts the instant answer at show time slightly better, because the expected effort of the card becomes an explicit input instead of something the state must carry.
- Cheapest decisive test: a feature-discovery style screen (as in feature discovery (SM70)) with the imputed duration as a candidate input on saved predictions, CPU, about an hour; only on a clear signal, a show-time instant-only cool-down arm against its control (about 3 GPU hours plus scoring).
- Expected win: small; the prior is low because the slow-Good signal check (SM88) and the retrieval-effort probe (SM122) show the model already uses past answer times.
- Risks: a new deploy input needs an exact online rule and a parity check in the fork.
- Rules: the accuracy go rule.

### Idea 11. A flatness-seeking cool-down (sharpness-aware minimisation), as a pointer from file names only

- Pattern: P17 (a module name in their manifest, method and result unknown) plus the general finding that flatter minima tolerate weight and state perturbations better.
- Hypothesis: a Small two-head cool-down with sharpness-aware minimisation lowers the rank-2 (Q2) format cost and possibly LogLoss at int8, without the training-loss term that hurt in the tail-energy penalty cool-down (SM149b).
- Cheapest decisive test: one arm like the control of the rank-2 cool-down (SM149) (about 39 minutes, about twice that with the extra gradient), scored at int8 and rank 2 on the 732 set (about an hour).
- Expected win: small, uncertain.
- Risks: the evidence from outside is a file name; doubled training cost; may simply tie.
- Rules: the accuracy go rule for the model, the format budgets for the format cost.

### Idea 12. Three-way input parity for the fork's feature builder (no GPU)

- Pattern: P11 (a deploy builder is right if and only if its vector equals the training row for the same review; their check found two drifted columns that engine-level parity could not see).
- Hypothesis: our engine-level parity (native against PyTorch on exported traces) cannot detect a drift between the fork's own feature construction from an Anki collection and the dataset rows the models were trained on; a row-by-row comparison would either confirm that the fork's inputs are exact or find a silent cost.
- Cheapest decisive test: first check whether such a check exists already (the fork review notes may contain one); if not, build a throwaway collection from a few benchmark users' review logs in a temporary folder (never from a real Anki profile), run the fork's feature builder on it, and compare every column with the dataset row of the same review. CPU only, about a day.
- Expected win: none if it is clean, potentially several tenths of a percent for users if a column drifted.
- Risks: driving the fork's builder from a synthetic collection is fiddly; deleted cards and filtered decks need the special cases the outside research lists.
- Rules: no reads under the Anki profile folder; benchmark users only.

### Idea 13. Query-batch size in the fork against the cache knee (CPU timing only)

- Pattern: P12 (their batched query path peaked at a batch of 128, about 11,300 reviews per second, and fell to about 8,200 at 512, while thread count made no difference).
- Hypothesis: the fork's due-queue scoring batch (their notes say the fork's default was 512) sits past the cache knee for our models too; a smaller batch, single-threaded, gives faster queue scoring and frees cores.
- Cheapest decisive test: a sweep of the batch size from 32 to 1,024 for each size in the fork's own benchmark, paired simultaneous trials with a one-sided Wilcoxon test, in a quiet window with `--timing`. A few CPU hours, not alongside GPU timing runs.
- Expected win: 10% to 40% faster queue scoring (their ratio was 1.38 times between 128 and 512), larger for Big.
- Risks: the knee depends on the CPU's caches and on the model size, so it must be measured on the user's machines; the state build (the larger cost in the fork speed test (SM111)) is not affected.
- Rules: an exact change (outputs identical), judged by timing only.

### Idea 14. Sensitivity-guided head pruning in the deck, preset and user streams

- Pattern: P1 and P5 (measure the structure before cutting) with the empirical Fisher or ablation saliency of each head as the tool.
- Observation: whole-layer cuts were blind: pruning (SM15) cut the context streams to 2 + 1 + 1 layers and cost +0.36% / +0.42%; the six-layer Tiny (SM130) took the same cut for Tiny. No head-level saliency was measured.
- Hypothesis: a few heads in the context streams carry most of their signal; removing the lowest-saliency heads (or narrowing those layers) costs less than a layer cut at the same parameter saving, which could buy back part of Tiny's size cost against Big or make Small cheaper per review.
- Cheapest decisive test: per-head ablation on the 8 probe users on the CPU (one head zeroed at a time, all six models, a few CPU hours); only if the saliency is concentrated, one cut plus cool-down on the GPU (about 5 GPU hours as the pruning design estimated).
- Expected win: small to medium, mostly on speed and parameters, which matter less than state for the phone.
- Risks: the per-review time is set by sequential layer steps rather than heads, so head cuts save less time than layer cuts.
- Rules: the Tiny and Small size budgets against Big.

### Idea 15. Format-aware post-training correction of a few output parameters (CPU)

- Pattern: P7 (a model trained with its format makes the format in-distribution), without training the whole model, using the bias-correction and output-reconstruction tools of post-training quantization.
- Hypothesis: least-squares refits of the per-head output group-norm gains and biases (a few hundred numbers per model) on training users, to make the rank-2 read output match the int8 read output, recover part of the rank-2 cost at zero deploy cost and without changing the int8 behaviour of the shipped weights (the corrected parameters ship only with the rank-2 format).
- Cheapest decisive test: dumps of the per-head read outputs at int8 and rank 2 on about 200 training users (CPU), the closed-form fit, then the 8 probe users on the CPU; GPU 732 only if at least a quarter of the rank-2 cost disappears.
- Expected win: 10% to 25% of the rank-2 cost (a guess; the full rank-2 cool-down (SM149) recovered a third with all weights free).
- Risks: small gain; it is not the rejected rank-2 QAT, but it is close in spirit and should be presented to the user as such.
- Rules: the 11-23 budgets.

## 4. Ranked shortlist

1. **Exact factored rank-(r+1) update (idea 1), CPU only, first.** It is exact (checked at 3e-14 on synthetic states), turns the low-rank speed warning into a likely gain for rank-2 states in the fork (SM151), and makes every later low-rank test cheaper to score, including ideas 2, 3 and 4.
2. **Output-aware truncation (idea 2), CPU pointer first, then one GPU pass.** The cheapest lever on the accuracy of the format being shipped, aimed at Big's thin rank-2 margin, with no training and a few KB of extra model data.
3. **Recurrent-mode long-card scorer (idea 6), CPU only, in parallel with 1 and 2.** It removes the coverage and power shortfall behind the long-card verdicts of the long-card state drift check (SM136), the trained smaller state for Tiny (SM137) and the low-rank int8 states test (SM148).
4. **Answer-time head shrinkage (idea 7), CPU only, under an hour.** A measured over-dispersion of 1.7 times in answer-time expressiveness (SM142) with an obvious statistical fix, on data we already have.
5. **Rate-distortion allocation per head, with energy-triggered growth (ideas 3 and 4), CPU step first, GPU after smaller shift vectors and factors on T24 states (SM152) reports.** The structure measured in the low-rank structure probe (SM147) says uniform ranks waste bytes; this is the pattern that gave the outside research its smallest passing formats.

Recommended order: ideas 1, 6 and 7 start at once on the CPU (none needs the GPU, so they can run beside GPU tickets); idea 2's CPU pointer follows idea 1, because the factored path makes its scoring cheaper; idea 3 starts with its free step 0 (the rate-distortion table from existing rows) and takes the GPU only after smaller shift vectors and factors on T24 states (SM152) has reported.
