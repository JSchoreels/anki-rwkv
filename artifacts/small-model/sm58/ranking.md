# SM58 follow-up 3: does the small model change the order of cards?

Per user and day: cards reviewed that day after >= 1 day away (first review of the card that day), days with >= 10 cards; pairs from at most 400 cards per day (seed 0). R = predicted recall, w8+s8, small vs fork big model.

| Model | Head | Users | Days | Cards | Kendall tau median (p10) | Spearman median | Pairs flipped | Risk top-10% overlap | Flipped pairs with one forgotten | Small model right |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k model (SM55) | curve | 722 | 191,902 | 19,149,155 | 0.841 (0.677) | 0.958 | 9.7% | 79.5% | 21,432,212 | 48.7% |
| 8k model (SM55) | instant | 722 | 191,902 | 19,149,155 | 0.850 (0.723) | 0.963 | 8.3% | 81.9% | 17,438,557 | 47.2% |
| Muon version (SM36) | curve | 722 | 191,902 | 19,149,155 | 0.837 (0.673) | 0.957 | 9.9% | 79.0% | 21,943,669 | 48.5% |
| Muon version (SM36) | instant | 722 | 191,902 | 19,149,155 | 0.846 (0.718) | 0.961 | 8.5% | 81.5% | 17,897,986 | 46.8% |

## Flipped pairs by how far apart the big model puts the two cards

| Model | Head | Big-model gap | Share of pairs | Flipped in that gap | Share of all flips |
|---|---|---|---:|---:|---:|
| 8k model (SM55) | curve | 0%-1% | 19.1% | 29.3% | 57.6% |
| 8k model (SM55) | curve | 1%-5% | 30.5% | 10.8% | 33.9% |
| 8k model (SM55) | curve | 5%-10% | 19.3% | 3.1% | 6.0% |
| 8k model (SM55) | curve | >= 10% | 31.1% | 0.8% | 2.5% |
| 8k model (SM55) | instant | 0%-1% | 24.3% | 22.2% | 64.7% |
| 8k model (SM55) | instant | 1%-5% | 27.3% | 9.0% | 29.4% |
| 8k model (SM55) | instant | 5%-10% | 17.0% | 2.3% | 4.7% |
| 8k model (SM55) | instant | >= 10% | 31.3% | 0.3% | 1.2% |
| Muon version (SM36) | curve | 0%-1% | 19.1% | 29.5% | 57.0% |
| Muon version (SM36) | curve | 1%-5% | 30.5% | 11.1% | 34.2% |
| Muon version (SM36) | curve | 5%-10% | 19.3% | 3.2% | 6.3% |
| Muon version (SM36) | curve | >= 10% | 31.1% | 0.8% | 2.6% |
| Muon version (SM36) | instant | 0%-1% | 24.3% | 22.4% | 64.0% |
| Muon version (SM36) | instant | 1%-5% | 27.3% | 9.3% | 29.8% |
| Muon version (SM36) | instant | 5%-10% | 17.0% | 2.5% | 4.9% |
| Muon version (SM36) | instant | >= 10% | 31.3% | 0.4% | 1.3% |
