# Energy-triggered rank growth (SM159): design

Written Tue 11-24, before any probe row of an energy label was scored or read. Branch `sm159`, worktree `C:/Users/jonat/workspace/wt/sm159`. The branch was created from `origin/main` and then reset to `origin/sm157` (4b2a86b6), because the low-rank codec (`scripts/state_lowrank.py` with the int8 states test (SM148), the int6 factors of smaller shift vectors and factors on T24 states (SM152) and the output-aware truncation pointer (SM157)) is not on `main`; `sm157` also holds the probe rows used for the bit-identity proof. CPU only, no GPU ticket, no queue, no training. Idea 4 of `wt/ideas-1123/artifacts/ideas-1123/OPPORTUNITIES.md`.

## Question

T24 (`w8+lr2.16.4q8x8`, the format the user chose for every size) stores every card and note head at rank 2 below 16 answers and at rank 4 from 16 answers: an age trigger. Instead, every (entity, layer, head) starts at rank 2 and grows by one rank, up to 4, only when truncation at its current rank would discard more than a share tau of the state's energy; the rank never shrinks. At the same mean state bytes as T24 on the probe users, does it cost less than T24? At T24's cost, does it use fewer bytes? The same question on top of int6 factors (F6, `w8+lr2.16.4q6x8`).

## Codec (`scripts/state_lowrank.py`, SM159 block)

- Label suffix `g<max>e<code>` after the shift width of a single-rank SM148 label: `w8+lr2q8x8g4e1500` = base rank 2, maximum rank 4, tau = code / 10,000 = 0.15, int8 factors, int8 shift vectors; `w8+lr2q6x8g4e1500` the same with int6 factors. Tiers, the R tier and the output-aware weights are refused with a growth suffix.
- Each (entity, layer, head) keeps a rank cap, 2 at the start. At each write, the factored solve of SM148 already gives the eigenvalues of `B B^T` (the squared singular values of the exact update S', which has rank at most cap + 1). Discarded share = 1 - (sum of the cap largest squared singular values) / (sum of all). When it is above tau and the cap is below 4, the cap grows by one at this write. The head is stored at rank min(n, cap) with the same factor format as SM148 / SM152 (left and right factor vectors, int8 or int6 codes with a float16 scale each). So a growing write is stored without truncation loss (only factor rounding). Non-finite heads keep their cap and are stored as they are (SM147 / SM148 rule).
- The fork's state record needs the cap: 2 bits per (layer, head), counted in the bytes below.
- Existing labels: their code paths are unchanged (`encode` is not touched; the growth path is a separate `encode_energy`; `lowrank_factors` only gains an optional return of the eigenvalues). New: every rank-only SM148 codec row gets an aggregate `lowrank_ranks` field (the end-of-history rank read-out: per stream, entity counts, the sum of stored ranks, rank histograms per head and per entity; no ids). The proof below checks every other field.

## Bytes (`sm159_bytes.py`)

The fork's compact layout of SM152 (`sm152_bytes.parts`, bit-packed codes): header (magic, precision byte, layer count, three u32 lengths per layer), shift vectors (int8, float16 scale each), WKV per layer and head at stored rank c: 2c factor vectors, each a float16 scale + ceil(K x bits / 8) bytes. Rank field: 1 byte per entity for T24, Q2 and F6 (as SM148 / SM152); for the energy labels ceil(layers x heads x 2 / 8) bytes per entity (Big card 3, Big note 2, Small card 2, Small note 1, six-layer Tiny 1). REF: K rows of a float16 scale + K int8 codes. Totals over every answered card and note of the users, at the end of their history (the stored ranks of the read-out). T24, Q2 and F6 bytes follow from the answer counts of the same read-out (min(n, 2) below 16 answers, 4 from 16), so every label is compared on exactly the same entities. The checks reproduce SM148's per-entity sizes. The int8 reference (REF) is the denominator of "bytes vs int8"; `rho = bytes / bytes(T24) - 1` (for int6 labels against F6).

## Calibration (`sm159_calib.py`, training users only, bytes only)

- Users: 20 training users (`users-calib.json`, `random.Random(159).sample` of the 100 users of the output-aware truncation fit (SM157) `users-fit.json`: training users of the 8k split with id >= 5000 and 4,000 to 8,000 reviews, none in any validation, test or evaluation pool; checked again by the script), 124,984 reviews.
- Forward passes of the scorer's CPU model (w8 weights, show-time inputs, leak-free, the scorer's `LeakBatches`), one per user and label; the loss is discarded and never read. Grid TRAIN_GRID = codes 800, 1000, 1250, 1500, 1800, 2200 (tau 0.08 to 0.22), int8 factors, every model.
- Matched code per model (`sm159_report.py select`): rho(code) on the 20 users; the code at rho = 0 by log-linear interpolation between the first adjacent grid pair (from the smallest code) where rho changes sign, rounded to an integer. If rho does not change sign the nearest grid end is used and flagged. This is EM (int8 factors) and, with the same code, EM6 (int6 factors).
- Grid range chosen on 2 other training users (`users-explore.json`, not in the calibration set, never a probe user), on Small two-head (SM118), six-layer Tiny two-head (SM130) and Big two-head (SM91s2): rho crossed 0 at about tau 0.15, 0.13 and 0.18. On those users T24 costs only 2.4% more bytes than Q2 (2.7% of their cards have 16 answers or more), so the growth budget is small and tau is high: at tau 0.15 about 6% of Small card heads grow to rank 3, mostly young entities.

## Labels scored on the probe users

| Name | Label | Meaning |
|---|---|---|
| REF | `w8+s8hx8` | int8 card and note rows (the reference of every cost) |
| Q2 | `w8+lr2q8x8` | rank 2 everywhere (proof label, and the lower end of the bytes) |
| T24 | `w8+lr2.16.4q8x8` | the shipped format (the comparison) |
| F6 | `w8+lr2.16.4q6x8` | T24 with int6 factors (the comparison of the int6 analog) |
| E500, E1000, E1500, E2000 | `w8+lr2q8x8g4e<code>` | fixed thresholds tau 0.05, 0.10, 0.15, 0.20, every model |
| EM | `w8+lr2q8x8g4e<matched code>` | matched on the training users, per model |
| EM6 | `w8+lr2q6x8g4e<matched code>` | the int6 analog at the same code |
| EMP, EMP6 | as EM / EM6 | fallback only (below) |

The fixed thresholds are meant to bracket T24's bytes on the probe users: those users include five long-card users, so T24 has more bytes there than on the training users, and the lower thresholds (0.05, 0.10) give the energy labels more room; tau 0.20 is close to Q2's bytes on the exploration users.

Fallback (`sm159_report.py fallback`, bytes only, before the report): if EM's probe bytes differ from T24's by more than 2% for a model, a probe-matched code is interpolated (the same rule as select) from the probe bytes of that model's scored int8 energy labels (the four fixed ones and EM); EMP and EMP6 are scored at that code for that model. Only the `lowrank_ranks` byte field of the rows is read for this step, never a LogLoss.

## Statistic

Cost of a label per (model, head) = 100 x (sum over the 8 users of its benchmark LogLoss x size / the same sum for REF - 1), review-weighted; paired user bootstrap, 2,000 draws, seed 0, 95% CI (as the output-aware truncation pointer (SM157)). Cells: curve and instant for Big two-head (SM91s2), Small two-head (SM118) and six-layer Tiny two-head (SM130), instant for Big Instant-Only (SM99), Small Instant-Only (SM113) and six-layer Tiny Instant-Only (SM130): 9 cells. Paired comparisons of an energy label against T24 (int6 labels against F6) use the same formula with T24 (F6) as the denominator.

Matched energy label per model and format (int8 vs T24; int6 vs F6): among EM and EMP (EM6 and EMP6), the one within 2% of the reference format's probe bytes with the smallest |rho|. If none is within 2%: the linear interpolation in bytes between the two scored energy labels of that factor format nearest to the reference bytes on each side, `lam x cost(a) + (1 - lam) x cost(b)` with `lam` from the bytes, bootstrapped with the same draws. If no pair brackets the reference bytes, the model is unmatched for that format.

## Rule (pre-registered, `sm159_report.py`; its selftest passes)

For each format (int8 factors vs T24; int6 factors vs F6), separately:

- Primary cells: for each of the three two-head models, the head where the reference format's (T24's, or F6's) cost is largest. A primary cell is informative when that cost is at least 0.01%.
- PASS when (a) every model is matched, (b) at least 2 of the 3 primary cells are informative, (c) on every informative primary cell the matched energy label's cost is at most 80% of the reference format's cost (a reduction of at least 20% of the reference format's cost), and (d) on none of the 9 cells is the matched energy label worse than the reference format by more than +0.01% (direct paired %, point value).
- Not (a) or not (b): inconclusive for that format.
- **GO** for a 732-set GPU run if the int8 comparison or the int6 comparison passes. The GPU run would then be its own pre-registered design (not prepared here), judged by the 11-23 budgets. Eight users make this a pointer, not a gate.
- Proof failure (below) means no verdict until fixed.

Reported outside the rule: every label's cost against REF with CIs; bytes vs int8 and vs T24 / F6 per label and model, card and note means per answered entity; the rank distribution (share of answered entities by their largest head rank 2, 3, 4, cards and notes, overall and for entities with 16 or more answers; share of heads at each rank); "at T24's cost, fewer bytes": per model the scored int8 energy label with the fewest bytes whose point cost is at most T24's on every head of the model.

## Proofs

- K1 (`cpu/selftest.out`): `state_lowrank` selftests of SM147, SM148, SM157 unchanged, plus SM159: the discarded share equals the float64 SVD tail share; caps start at 2, grow by at most one per write, never shrink and stay at most 4; a growing write stores the exact update (float factors); tau above 1 never grows and equals plain rank 2 (float factors to 2e-5; int8 factors identical on 98% to 99% of heads, the rest differ by a flipped int8 code from float noise, as the discrete-codec rule of SM150 allows); grammar and descriptors; the rank read-out over groups of entities. `sm159_bytes.py` reproduces SM148's entity sizes; `sm159_report.py selftest` checks the rule on synthetic rows.
- K2 (in the report): REF, Q2 and T24 on the 8 probe users x 6 models (144 rows) equal the committed SM157 probe rows (`../sm157/cpu/sm157-rows.jsonl`) in every field except timing, provenance and the new `lowrank_ranks` field (identical JSON values, not a tolerance).
- K3 (in the report): the T24 and Q2 rows' read-out (stored ranks of the codec) equals the answer-count formula used for the T24, Q2 and F6 bytes.
- A pipeline check before this design was committed: user 2388 x six-layer Tiny two-head (SM130) (`g4e1500` int8 and int6, T24) and Big two-head (SM91s2) (`g4e1500`): rows finite, descriptors and read-out present with the right layers (Tiny card and note 1 layer, Big card 3 and note 2). Only these properties were looked at.

## Run (`run-sm159.sh`, detached)

Step 0 selftests; step 1 calibration (three processes: Big two-head; Big Instant-Only; Small and Tiny), merged into `cpu/calib-train.json` (aggregates only); step 2 select -> `sm159-taus.json`; step 3 the 8 probe users x 6 models x the probe labels, three processes (same split) -> `cpu/sm159-rows.jsonl`; step 4 fallback (bytes only) and its extra labels; step 5 `sm159_report.py report` -> `sm159-result.{json,md}`. Three CPU processes at a time, 3 threads each (a GPU job with CPU data workers is running), memory guard at 5.9 GB per process (`memguard.ps1`; probe scoring used about 2.3 GB in SM157). Expected about 1.5 hours.

## Files

`design.md`, `models.sh` (SM157's copy), `users-probe.json` (SM157's), `users-calib.json`, `users-explore.json`, `sm159_calib.py`, `sm159_bytes.py`, `sm159_report.py`, `run-sm159.sh`, `memguard.ps1`; outputs `cpu/` (selftest, calibration read-outs, rows, logs), `sm159-taus.json`, `sm159-result.{json,md}`, then `RESULT.md` and `HANDOFF.md`.
