| Case | Status | Parameters | Reviews/s | vs w32 | Training s | Updates | Main CPU s | Machine CPU s | Peak MiB | Overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| w32 | failed | — | — | — | — | — | 10 | 36 | — | — |
| w80-b256-wide | complete | 558209 | 111,281 | -37.6% | 111.5 | 466 | 145 | 827 | 8417 | 100% |
| w64-b256-wide | complete | 331511 | 141,274 | -20.7% | 87.8 | 466 | 117 | 756 | 6687 | 100% |
| w80-b256 | complete | 558209 | 62,216 | -65.1% | 199.4 | 466 | 204 | 1048 | 8530 | 100% |
| w64-b256 | complete | 331511 | 74,584 | -58.2% | 166.3 | 466 | 133 | 802 | 6778 | 21% |
| w80-sequential-b256-wide | complete | 558209 | 119,808 | -32.8% | 103.6 | 466 | 128 | 583 | 8399 | 0% |
| w64-wide | complete | 331511 | 39,041 | -78.1% | 317.8 | 287 | 151 | 1357 | 13042 | 0% |

w32 control missing or failed in this run: vs w32 uses the accepted 178,242 reviews/s (docs/GPU_OPTIMIZATIONS.md).
