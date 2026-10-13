# Historical best configuration versus today's control

The human asked whether today's roughly 166k reviews/s control was a weaker configuration than the historical roughly 190–197k run. The saved records establish that it is the same configuration: options and config dictionaries are identical, both process 7,654,975 reviews in 167 updates over 80 training and 20 evaluation users, and initial/final parameter hashes and held-out input hashes match exactly. Torch 2.9.1+cu128 and RTX 5080 identifiers also match. [Reproducible audit](analyze.py) and [raw comparison/provenance](comparison.json) retain the evidence; no new GPU run was launched for this audit.

| Recorded quantity | Historical 0042 | Current 0072 control r1 |
|---|---:|---:|
| Reviews/s | 197,499 | 166,338 |
| Training seconds | 38.759 | 46.021 |
| Queue wait seconds | 0.472 | 0.905 |
| Summed preparation seconds across overlapping jobs | 118.276 | 162.951 |
| GPU copy event span seconds | 2.279 | 3.107 |
| Garbage collection seconds | 0.140 | 0.169 |
| Discarded primer seconds | 15.311 | 15.356 |
| Observed training GPU utilization mean | 66.54% | 60.94% |
| Observed worker-pool CPU mean | 53.23% | 62.27% |
| Validation seconds | 29.626 | 32.447 |

The 7.261-second training gap is not explained by the 0.433-second queue-wait increase or the 0.030-second GC increase. Summed preparation is overlapping worker wall time and cannot be subtracted from serial training time. GPU copy event spans can include dispatch/device idle gaps; they are not a direct measurement of PCIe bandwidth. Resource means summarize the recorded samples, whose windows are shorter than the general 120-second resource-assessment minimum; they are descriptive drift evidence, not a causal diagnosis or promotion gate.

The implementation and harness have evolved. The archived source diff adds per-update CUDA events and elapsed-time reads, disabled native/experimental branches, and expanded GPU state sampling. [Training source diff](training-source.diff), [backend/model/monitor source diffs](additional-source.diff), and [prefetch source diff](prefetch-source.diff) compare the preserved source archives, not inferred historical HEADs. The thread-limit helper already exists in the historical source; complete historical thread environment and GPU clocks were not recorded, so a changed thread setting or lower clock is unproven. The captured current kernel's 18.176 Kbyte/block includes 17.152 static plus 1.024 driver-reserved shared memory; this does not establish an increase over the prior 17,152-byte compiled static allocation.

Contemporaneous paired runs test whether the native candidate improves the best configuration under today's common source and machine conditions. Native WKV passed at +1.81426% paired-update mean and +0.38273% lower bound with exact parameters and four losses; that is a local improvement, not evidence that the historical 197,499 reviews/s record has been exceeded. Preserve both facts. The absolute-baseline gap remains unresolved.

Pending R01-M01 follow-up: measure per-update CUDA event creation/recording and expanded telemetry overhead with a matched control, preserve the existing host-wall timing estimator and calibrated gate, and audit the default CUDA backend's compiled behavior. Reusing preinitialized event pairs is a possible exact implementation track, but gains are unmeasured. Removing device spans would require explicit handling of missing diagnostics in the loader rather than inventing zero measurements. No administrator prompts or new validation cohort are needed for a later scoped comparison; it must not overlap current GPU timing jobs.
