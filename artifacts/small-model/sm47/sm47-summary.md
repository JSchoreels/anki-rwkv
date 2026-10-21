# SM47: float16 row scales (h) and int8 shift vectors (x8), SM36 (best model), w8 weights

State bits per entity: per WKV row 16 codes + one scale (float32, or float16 with h); per token-shift vector 64 float32, or with x8 64 int8 codes + one scale (float16 with h, else float32). Card = 2 WKV layers x 4 heads x 16 rows + 3 shift vectors; note = 1 x 4 x 16 rows + 2 shift vectors. Collection = 13,973 cards + 13,198 notes (MB = 10^6 bytes).

| Variant | Card bits | Note bits | Collection MB | vs w8+s8 |
|---|---:|---:|---:|---:|
| w8+s8 | 26,624 | 14,336 | 70.15 | +0.0% |
| w8+s4c8n | 18,432 | 14,336 | 55.84 | -20.4% |
| w8+s8h | 24,576 | 13,312 | 64.89 | -7.5% |
| w8+s8x8 | 22,112 | 11,328 | 57.31 | -18.3% |
| w8+s8hx8 | 20,016 | 10,272 | 51.91 | -26.0% |
| w8+s4c8nh | 16,384 | 13,312 | 50.58 | -27.9% |
| w8+s4c8nx8 | 13,920 | 11,328 | 43.00 | -38.7% |
| w8+s4c8nhx8 | 11,824 | 10,272 | 37.60 | -46.4% |

Review-weighted benchmark LogLoss; d s8 = variant - SM36 w8+s8, d base = variant - its base (w8+s8 or SM45 w8+s4c8n), [95% paired user bootstrap]; % vs the big model (fork w8+s8) on the same users; curve = ahead.

- 732 pool, w8+s8h (base w8+s8; 24,576 / 13,312 bits, 64.89 MB), 723 users: curve 0.30333 (+0.64% vs big; s8 0.30333 +0.64%) d s8 +0.00000 [-0.00000, +0.00000]; instant 0.27960 (+0.63% vs big; s8 0.27960 +0.63%) d s8 +0.00000 [-0.00000, +0.00000]
- 732 pool, w8+s8x8 (base w8+s8; 22,112 / 11,328 bits, 57.31 MB), 723 users: curve 0.30333 (+0.64% vs big; s8 0.30333 +0.64%) d s8 -0.00000 [-0.00000, +0.00000]; instant 0.27960 (+0.63% vs big; s8 0.27960 +0.63%) d s8 +0.00000 [-0.00000, +0.00000]
- 732 pool, w8+s8hx8 (base w8+s8; 20,016 / 10,272 bits, 51.91 MB), 723 users: curve 0.30333 (+0.64% vs big; s8 0.30333 +0.64%) d s8 -0.00000 [-0.00000, +0.00000]; instant 0.27960 (+0.63% vs big; s8 0.27960 +0.63%) d s8 +0.00000 [-0.00000, +0.00000]
- 732 pool, w8+s4c8nh (base w8+s4c8n; 16,384 / 13,312 bits, 50.58 MB), 723 users: curve 0.30338 (+0.66% vs big; s8 0.30333 +0.64%) d s8 +0.00005 [+0.00003, +0.00006]; d base -0.00000 [-0.00000, +0.00000]; instant 0.27965 (+0.64% vs big; s8 0.27960 +0.63%) d s8 +0.00005 [+0.00004, +0.00006]; d base +0.00000 [-0.00000, +0.00000]
- 732 pool, w8+s4c8nx8 (base w8+s4c8n; 13,920 / 11,328 bits, 43.00 MB), 723 users: curve 0.30338 (+0.66% vs big; s8 0.30333 +0.64%) d s8 +0.00005 [+0.00003, +0.00006]; d base +0.00000 [-0.00000, +0.00000]; instant 0.27965 (+0.64% vs big; s8 0.27960 +0.63%) d s8 +0.00005 [+0.00004, +0.00006]; d base -0.00000 [-0.00000, +0.00000]
- 732 pool, w8+s4c8nhx8 (base w8+s4c8n; 11,824 / 10,272 bits, 37.60 MB), 723 users: curve 0.30338 (+0.66% vs big; s8 0.30333 +0.64%) d s8 +0.00005 [+0.00003, +0.00006]; d base +0.00000 [-0.00000, +0.00000]; instant 0.27965 (+0.64% vs big; s8 0.27960 +0.63%) d s8 +0.00005 [+0.00004, +0.00006]; d base -0.00000 [-0.00000, +0.00000]
- 464 pool, w8+s8h (base w8+s8; 24,576 / 13,312 bits, 64.89 MB), 458 users: curve 0.29804 (+0.74% vs big; s8 0.29804 +0.74%) d s8 +0.00000 [-0.00000, +0.00000]; instant 0.27397 (+0.64% vs big; s8 0.27397 +0.64%) d s8 +0.00000 [-0.00000, +0.00000]
- 464 pool, w8+s8x8 (base w8+s8; 22,112 / 11,328 bits, 57.31 MB), 458 users: curve 0.29804 (+0.74% vs big; s8 0.29804 +0.74%) d s8 -0.00000 [-0.00000, +0.00000]; instant 0.27397 (+0.64% vs big; s8 0.27397 +0.64%) d s8 +0.00000 [-0.00000, +0.00000]
- 464 pool, w8+s8hx8 (base w8+s8; 20,016 / 10,272 bits, 51.91 MB), 458 users: curve 0.29804 (+0.74% vs big; s8 0.29804 +0.74%) d s8 -0.00000 [-0.00000, +0.00000]; instant 0.27397 (+0.64% vs big; s8 0.27397 +0.64%) d s8 +0.00000 [-0.00000, +0.00000]
- 464 pool, w8+s4c8nh (base w8+s4c8n; 16,384 / 13,312 bits, 50.58 MB), 458 users: curve 0.29809 (+0.76% vs big; s8 0.29804 +0.74%) d s8 +0.00004 [+0.00003, +0.00007]; d base -0.00000 [-0.00000, +0.00000]; instant 0.27402 (+0.66% vs big; s8 0.27397 +0.64%) d s8 +0.00005 [+0.00003, +0.00007]; d base -0.00000 [-0.00000, +0.00000]
- 464 pool, w8+s4c8nx8 (base w8+s4c8n; 13,920 / 11,328 bits, 43.00 MB), 458 users: curve 0.29809 (+0.76% vs big; s8 0.29804 +0.74%) d s8 +0.00004 [+0.00003, +0.00007]; d base -0.00000 [-0.00000, +0.00000]; instant 0.27402 (+0.66% vs big; s8 0.27397 +0.64%) d s8 +0.00005 [+0.00003, +0.00006]; d base -0.00000 [-0.00000, +0.00000]
- 464 pool, w8+s4c8nhx8 (base w8+s4c8n; 11,824 / 10,272 bits, 37.60 MB), 458 users: curve 0.29809 (+0.76% vs big; s8 0.29804 +0.74%) d s8 +0.00004 [+0.00003, +0.00007]; d base -0.00000 [-0.00000, +0.00000]; instant 0.27402 (+0.66% vs big; s8 0.27397 +0.64%) d s8 +0.00005 [+0.00003, +0.00006]; d base -0.00000 [-0.00000, +0.00000]
