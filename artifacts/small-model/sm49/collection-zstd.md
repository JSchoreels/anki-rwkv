# SM49: lossless compression of the stored int state codes, own collection (aggregate only)

SM36 states after the whole history of the 2026-10-14 collection copy (read-only); MB = 10^6 bytes, cards + notes; ratio = raw / compressed. zstd = libzstd 1.5.7 (Python 3.15 compression.zstd).

| Variant | Entities (cards / notes) | Raw MB | zstd-3 columnar | zstd-19 columnar | zstd-22 columnar | zstd-19 records | per-entity zstd-19 | per-entity zstd-3 + 64 KB dict | order-0 entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w8+s4c8nhx8 | 13,953 / 13,178 | 37.54 | 32.59 (1.15x) | 32.15 (1.17x) | 32.15 (1.17x) | 33.45 (1.12x) | 36.20 (1.04x) | 36.01 (1.04x) | 32.90 (1.14x) |
| w8+s4hgx4 | 13,953 / 13,178 | 23.68 | 14.56 (1.63x) | 13.58 (1.74x) | 13.58 (1.74x) | 13.97 (1.70x) | 17.08 (1.39x) | 16.07 (1.47x) | 14.80 (1.60x) |

Order-0 entropy per field (bits per symbol, raw -> entropy):

- w8+s4c8nhx8: card_wkv_codes 4 -> 3.41 (14.29 -> 12.16 MB); card_wkv_scales 16 -> 12.94 (3.57 -> 2.89 MB); card_shift_codes 8 -> 7.08 (2.68 -> 2.37 MB); card_shift_scales 16 -> 10.00 (0.08 -> 0.05 MB); note_wkv_codes 8 -> 7.44 (13.49 -> 12.55 MB); note_wkv_scales 16 -> 11.96 (1.69 -> 1.26 MB); note_shift_codes 8 -> 7.47 (1.69 -> 1.57 MB); note_shift_scales 16 -> 10.22 (0.05 -> 0.03 MB)
- w8+s4hgx4: card_wkv_codes 4 -> 2.37 (14.29 -> 8.45 MB); card_wkv_scales 16 -> 12.67 (0.22 -> 0.18 MB); card_shift_codes 4 -> 2.98 (1.34 -> 1.00 MB); card_shift_scales 16 -> 9.93 (0.08 -> 0.05 MB); note_wkv_codes 4 -> 2.55 (6.75 -> 4.29 MB); note_wkv_scales 16 -> 12.11 (0.11 -> 0.08 MB); note_shift_codes 4 -> 3.36 (0.84 -> 0.71 MB); note_shift_scales 16 -> 10.20 (0.05 -> 0.03 MB)

Same runs, collection LogLoss (aggregate, full layout, 173,690 benchmark reviews; Δ% vs fork 2.76M w8+s8 0.441652 / 0.436683 from wt/sm42/artifacts/small-model/collection/coll-fork.jsonl):

| Variant | curve LogLoss | Δ vs fork | instant LogLoss | Δ vs fork |
|---|---:|---:|---:|---:|
| w8+s4c8nhx8 (SM47) | 0.44139 | -0.06% | 0.43715 | +0.11% |
| w8+s4hgx4 (SM49 C) | 0.44301 | +0.31% | 0.43877 | +0.48% |

Dumps: collection_codes.py (stored state = the scorer's reconstruction, checked with torch.equal at every step), private scratch directory outside git; 13,953 cards / 13,178 notes have a review in the 2026-10-14 copy (the live collection has 13,973 / 13,198).
