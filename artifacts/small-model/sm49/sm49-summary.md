# SM49: smaller int states (int4 notes, int4 shift vectors, per-head scales), SM36 (best model), w8 weights

State bits per entity: per WKV row 16 codes + one float16 scale (g: one float16 scale per head of 16 rows instead); per token-shift vector 64 codes (x8 int8, x4 int4) + one float16 scale. Card = 2 WKV layers x 4 heads x 16 rows + 3 shift vectors; note = 1 x 4 x 16 rows + 2 shift vectors. Collection = 13,973 cards + 13,198 notes (MB = 10^6 bytes).

| Variant | Card bits | Note bits | Collection MB | vs w8+s8 |
|---|---:|---:|---:|---:|
| SM36 w8+s8 | 26,624 | 14,336 | 70.15 | +0.0% |
| SM47 w8+s4c8nhx8 | 11,824 | 10,272 | 37.60 | -46.4% |
| A w8+s4hx8 | 11,824 | 6,176 | 30.84 | -56.0% |
| B w8+s4hx4 | 11,056 | 5,664 | 28.65 | -59.2% |
| C w8+s4hgx4 | 9,136 | 4,704 | 23.72 | -66.2% |
| C' w8+s4hgx4c8hx8n | 9,136 | 10,272 | 32.90 | -53.1% |

Review-weighted benchmark LogLoss; d s8 = variant - SM36 w8+s8, d base = variant - its base, [95% paired user bootstrap]; % vs the big model (fork w8+s8) on the same users; curve = ahead.

- 732 pool, A w8+s4hx8 (base w8+s4c8nhx8; 11,824 / 6,176 bits, 30.84 MB), 723 users: curve 0.30343 (+0.68% vs big; s8 0.30333 +0.64%) d s8 +0.00010 [+0.00007, +0.00013]; d base +0.00005 [+0.00003, +0.00008]; instant 0.27976 (+0.69% vs big; s8 0.27960 +0.63%) d s8 +0.00017 [+0.00014, +0.00020]; d base +0.00012 [+0.00009, +0.00015]
- 732 pool, B w8+s4hx4 (base w8+s4hx8; 11,056 / 5,664 bits, 28.65 MB), 723 users: curve 0.30346 (+0.69% vs big; s8 0.30333 +0.64%) d s8 +0.00012 [+0.00009, +0.00016]; d base +0.00003 [+0.00000, +0.00005]; instant 0.27990 (+0.73% vs big; s8 0.27960 +0.63%) d s8 +0.00030 [+0.00027, +0.00034]; d base +0.00014 [+0.00012, +0.00015]
- 732 pool, C w8+s4hgx4 (base w8+s4hx4; 9,136 / 4,704 bits, 23.72 MB), 723 users: curve 0.30401 (+0.87% vs big; s8 0.30333 +0.64%) d s8 +0.00068 [+0.00054, +0.00083]; d base +0.00055 [+0.00045, +0.00068]; instant 0.28068 (+1.02% vs big; s8 0.27960 +0.63%) d s8 +0.00108 [+0.00094, +0.00124]; d base +0.00078 [+0.00067, +0.00090]
- 732 pool, C' w8+s4hgx4c8hx8n (base w8+s4c8nhx8; 9,136 / 10,272 bits, 32.90 MB), 723 users: curve 0.30361 (+0.74% vs big; s8 0.30333 +0.64%) d s8 +0.00028 [+0.00022, +0.00034]; d base +0.00023 [+0.00018, +0.00029]; instant 0.28002 (+0.78% vs big; s8 0.27960 +0.63%) d s8 +0.00042 [+0.00036, +0.00049]; d base +0.00037 [+0.00032, +0.00043]
- 464 pool, A w8+s4hx8 (base w8+s4c8nhx8; 11,824 / 6,176 bits, 30.84 MB), 458 users: curve 0.29814 (+0.78% vs big; s8 0.29804 +0.74%) d s8 +0.00009 [+0.00006, +0.00013]; d base +0.00005 [+0.00002, +0.00008]; instant 0.27413 (+0.70% vs big; s8 0.27397 +0.64%) d s8 +0.00016 [+0.00012, +0.00021]; d base +0.00011 [+0.00008, +0.00015]
- 464 pool, B w8+s4hx4 (base w8+s4hx8; 11,056 / 5,664 bits, 28.65 MB), 458 users: curve 0.29816 (+0.79% vs big; s8 0.29804 +0.74%) d s8 +0.00012 [+0.00008, +0.00016]; d base +0.00003 [-0.00001, +0.00005]; instant 0.27427 (+0.75% vs big; s8 0.27397 +0.64%) d s8 +0.00029 [+0.00025, +0.00035]; d base +0.00014 [+0.00012, +0.00015]
- 464 pool, C w8+s4hgx4 (base w8+s4hx4; 9,136 / 4,704 bits, 23.72 MB), 458 users: curve 0.29872 (+0.97% vs big; s8 0.29804 +0.74%) d s8 +0.00068 [+0.00051, +0.00086]; d base +0.00056 [+0.00042, +0.00071]; instant 0.27503 (+1.03% vs big; s8 0.27397 +0.64%) d s8 +0.00105 [+0.00087, +0.00126]; d base +0.00076 [+0.00062, +0.00091]
- 464 pool, C' w8+s4hgx4c8hx8n (base w8+s4c8nhx8; 9,136 / 10,272 bits, 32.90 MB), 458 users: curve 0.29832 (+0.84% vs big; s8 0.29804 +0.74%) d s8 +0.00028 [+0.00020, +0.00037]; d base +0.00023 [+0.00017, +0.00031]; instant 0.27439 (+0.79% vs big; s8 0.27397 +0.64%) d s8 +0.00041 [+0.00034, +0.00050]; d base +0.00037 [+0.00031, +0.00044]
