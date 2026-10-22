# SM58 follow-up: how much the small and big model disagree

x = p_small - p_big per review (w8+s8). Shares are of all test reviews in the sample; the central 5-point UM+ bin is |x| < 0.025. Own collection: aggregate only.

## Benchmark users (722 users)

| Model | Head | Sample | Reviews | |x| >= 0.025 | |x| >= 0.05 | |x| >= 0.10 | p66 |x| | p95 |x| | p99 |x| |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Muon version (SM36) | curve | all | 30,126,145 | 24.9% (7,497,828) | 10.0% | 2.5% | 0.0178 | 0.0729 | 0.1443 |
| Muon version (SM36) | curve | cross_day | 19,287,209 | 24.4% (4,709,382) | 9.9% | 2.7% | 0.0176 | 0.0738 | 0.1518 |
| Muon version (SM36) | instant | all | 30,126,145 | 21.4% (6,453,683) | 8.0% | 1.5% | 0.0148 | 0.0633 | 0.1129 |
| Muon version (SM36) | instant | cross_day | 19,287,209 | 19.7% (3,806,939) | 7.0% | 1.2% | 0.0139 | 0.0591 | 0.1071 |
| 8k model (SM55) | curve | all | 30,126,145 | 24.1% (7,268,820) | 9.5% | 2.3% | 0.0172 | 0.0706 | 0.1376 |
| 8k model (SM55) | curve | cross_day | 19,287,209 | 23.5% (4,537,105) | 9.3% | 2.4% | 0.0170 | 0.0711 | 0.1430 |
| 8k model (SM55) | instant | all | 30,126,145 | 20.8% (6,273,285) | 7.7% | 1.4% | 0.0144 | 0.0617 | 0.1100 |
| 8k model (SM55) | instant | cross_day | 19,287,209 | 19.1% (3,683,848) | 6.7% | 1.1% | 0.0135 | 0.0575 | 0.1044 |

## Own collection

| Model | Head | Sample | Reviews | |x| >= 0.025 | |x| >= 0.05 | |x| >= 0.10 | p66 |x| | p95 |x| | p99 |x| |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Muon version (SM36) | curve | all | 173,690 | 32.8% (56,977) | 12.8% | 2.5% | 0.0242 | 0.0790 | 0.1258 |
| Muon version (SM36) | curve | cross_day | 141,775 | 33.7% (47,752) | 13.0% | 2.4% | 0.0248 | 0.0793 | 0.1240 |
| Muon version (SM36) | instant | all | 173,690 | 30.8% (53,455) | 11.4% | 1.7% | 0.0228 | 0.0724 | 0.1141 |
| Muon version (SM36) | instant | cross_day | 141,775 | 31.3% (44,436) | 11.3% | 1.6% | 0.0232 | 0.0715 | 0.1110 |
| 8k model (SM55) | curve | all | 173,690 | 33.8% (58,681) | 14.1% | 3.1% | 0.0249 | 0.0845 | 0.1332 |
| 8k model (SM55) | curve | cross_day | 141,775 | 34.9% (49,514) | 14.6% | 3.2% | 0.0257 | 0.0858 | 0.1327 |
| 8k model (SM55) | instant | all | 173,690 | 30.3% (52,627) | 11.2% | 1.7% | 0.0224 | 0.0717 | 0.1142 |
| 8k model (SM55) | instant | cross_day | 141,775 | 30.7% (43,504) | 11.0% | 1.5% | 0.0227 | 0.0703 | 0.1104 |
