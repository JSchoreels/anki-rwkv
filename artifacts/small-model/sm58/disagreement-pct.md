# SM58 follow-up 2: percentiles of the R difference between the small and big model

x = R_small - R_big per review (w8+s8, R = predicted recall probability). Percentiles are of |x| (0.01 = 1 point); mean x > 0 = the small model predicts higher recall. Own collection: aggregate only.

## Benchmark users (722 users)

| Model | Head | Sample | Reviews | p50 | p90 | p95 | p99 | mean abs | mean x |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Muon version (SM36) | curve | all | 30,126,145 | 0.0099 | 0.0499 | 0.0729 | 0.1443 | 0.0201 | +0.0015 |
| Muon version (SM36) | curve | cross_day | 19,287,209 | 0.0100 | 0.0496 | 0.0738 | 0.1518 | 0.0203 | +0.0026 |
| Muon version (SM36) | instant | all | 30,126,145 | 0.0076 | 0.0441 | 0.0633 | 0.1129 | 0.0166 | +0.0006 |
| Muon version (SM36) | instant | cross_day | 19,287,209 | 0.0073 | 0.0409 | 0.0591 | 0.1071 | 0.0156 | +0.0006 |
| 8k model (SM55) | curve | all | 30,126,145 | 0.0096 | 0.0485 | 0.0706 | 0.1376 | 0.0194 | +0.0009 |
| 8k model (SM55) | curve | cross_day | 19,287,209 | 0.0096 | 0.0479 | 0.0711 | 0.1430 | 0.0195 | +0.0021 |
| 8k model (SM55) | instant | all | 30,126,145 | 0.0074 | 0.0430 | 0.0617 | 0.1100 | 0.0161 | +0.0009 |
| 8k model (SM55) | instant | cross_day | 19,287,209 | 0.0071 | 0.0398 | 0.0575 | 0.1044 | 0.0152 | +0.0010 |

## Own collection

| Model | Head | Sample | Reviews | p50 | p90 | p95 | p99 | mean abs | mean x |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Muon version (SM36) | curve | all | 173,690 | 0.0155 | 0.0573 | 0.0790 | 0.1258 | 0.0242 | +0.0049 |
| Muon version (SM36) | curve | cross_day | 141,775 | 0.0161 | 0.0578 | 0.0793 | 0.1240 | 0.0246 | +0.0065 |
| Muon version (SM36) | instant | all | 173,690 | 0.0146 | 0.0536 | 0.0724 | 0.1141 | 0.0226 | +0.0065 |
| Muon version (SM36) | instant | cross_day | 141,775 | 0.0151 | 0.0533 | 0.0715 | 0.1110 | 0.0227 | +0.0086 |
| 8k model (SM55) | curve | all | 173,690 | 0.0156 | 0.0614 | 0.0845 | 0.1332 | 0.0253 | +0.0064 |
| 8k model (SM55) | curve | cross_day | 141,775 | 0.0163 | 0.0627 | 0.0858 | 0.1327 | 0.0259 | +0.0084 |
| 8k model (SM55) | instant | all | 173,690 | 0.0144 | 0.0531 | 0.0717 | 0.1142 | 0.0223 | +0.0049 |
| 8k model (SM55) | instant | cross_day | 141,775 | 0.0147 | 0.0524 | 0.0703 | 0.1104 | 0.0223 | +0.0064 |
