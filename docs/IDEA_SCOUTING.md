# External idea scouting ledger

Ideas collected from outside the repository (papers, kernel libraries, vendor documentation, the upstream benchmark) and judged against the measured R01 facts. Every idea that is worth an experiment has a track row in [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) with the same ID; this file keeps the reasoning, the sources and the first concrete step so the overview stays short. Ideas that were surveyed and set aside are listed too, with the reason, so later scouting does not repeat them. Add a new dated pass at the end rather than rewriting earlier ones.

Exactness classes used below: **exact** means the candidate does not change any floating-point result and can be promoted through the ordinary R01 gate (bitwise-equal updates, measured gain above the timing-noise band); **non-exact** means it changes training results and needs the pending seed-replicated non-inferiority gate (P-B in [R01_NEXT_ROUND_PLAN.md](R01_NEXT_ROUND_PLAN.md)) or an explicit human decision; nothing here is authorized to run by itself. Architecture-changing experiments are allowed in principle (human decision 2026-10-13: "I don't mind re-architecturing tests, but let's keep all those in a ledger"), so each non-exact idea states what it would change and how it would be judged.

## Pass 1: 2026-10-13

### Repository facts the ideas were judged against

- Expertium model: `d_model` 32 with one head, five RWKV-7 streams run in sequence (card 1, deck 4, note 3, preset 3, global 3 = 14 layers), about 31 layer instances per update, about 60 small ops per time mixer; all modules are TorchScript with graph-executor optimization disabled ([R01_HOST_PATH.md](R01_HOST_PATH.md)).
- Per real batch about 8,284 kernel launches and 34,305 ATen dispatches; the raw NVRTC launcher costs about 12.5 µs self per call; summed host dispatch roughly equals summed GPU span, which fits host/GPU lockstep through command-buffer back-pressure ([idea scouting after 0070](R01_HOST_PATH.md#idea-scouting-after-0070-analysis-only-2026-10-13)).
- Updates below about 49k reviews have a nearly flat 169–193 ms GPU span (72 such updates carry 19.4% of reviews but 14.5 of 40.9 s summed GPU span); updates of at least 49k reviews run at 4.27 µs per review and are led by the WKV backward. Fixed per-update cost is bounded near 20% of training time.
- Windows 11 WDDM, RTX 5080 (sm_120, 16 GiB), PyTorch 2.9.1+cu128, CuPy 14.2, NVRTC-built exact kernels; the machine's scheduling state alone moves identical runs between about 162k and 199k reviews/s, so the promotion rule uses the measured timing-noise band.
- The R01 loss ceiling is 0.1% on one seed; the F22 fastmode comparison moved held-out losses by +0.84% to +1.48% on one seed, so float-order changes rarely pass without the seed-replicated gate.
- Upstream `open-spaced-repetition/srs-benchmark` keeps RWKV-Instant at the top of its table (log loss 0.2773 without same-day reviews, 0.2660 with; RWKV-Curve 0.3193 / 0.2974; FSRS-7 with recency weighting 0.3370; LSTM with 8,869 parameters 0.3332). RWKV is trained on 5,000 users and evaluated on the other 5,000, not tuned per user. Recent upstream `rwkv/` commits (Feb–Aug 2026) touch LMDB-per-worker reads, portable process priority, tensor-shape annotations and the FSRS-7/GRU re-benchmark; none changes the architecture or kernels.

### New tracks

#### F75 Persistent fused time-mixer kernel (one launch per layer instance)

- What: fuse the roughly 60 small ops of a time mixer (LayerNorm, shift gather, eight-way lerp, four 32×32 linears, LoRA projections, activations, the WKV recurrence, GroupNorm, bonus) into one persistent kernel per layer instance that keeps the 32×32 state and the 32-wide weights in registers/shared memory for the whole split; the channel mixer can follow the same pattern.
- Why here: with `d_model` 32 every op is a launch-bound micro-kernel; launch count is nearly independent of update size, which is exactly the regime where Persistent RNNs (weights kept on-chip across timesteps, 2.8 TFLOP/s at batch 4 on a TitanX) and FlashRNN (recurrent matmul and pointwise ops fused in one kernel inside the sequential loop, head-wise parallelism, state in registers/SRAM) report their largest gains. The small-update floor (169–193 ms) and the ~8.3k launches per batch are the measured headroom.
- Exactness: elementwise segments (LayerNorm, lerp, activations, GroupNorm, bonus) can reproduce PyTorch's order and rounding bitwise, as the F34/F32 exact fusions already do; the four linears and the recurrence cannot reproduce cuBLAS/accepted-kernel summation order without copying it, so the full fusion is non-exact unless each matmul segment copies the accepted kernel's arithmetic. Plan it in two stages: exact prologue/epilogue fusion first, full fusion under the seed gate.
- First step: per-layer-instance launch and dispatch attribution from the existing dispatch profile (counts only, not wall time), then a forward-only prologue kernel (LayerNorm + shift + lerp + LoRA activations) with the synthetic and real-batch bitwise checks the repository already uses.
- Risks: register pressure for the backward (FlashRNN notes the fused design has a hard head-size ceiling set by SRAM); a persistent kernel serializes the split's rows unless rows are mapped to blocks; Python re-entry per custom op (F65) must not eat the saved launches.
- Sources: [Persistent RNNs (Diamos et al., ICML 2016)](https://proceedings.mlr.press/v48/diamos16.html), [FlashRNN (ICLR 2025, arXiv 2412.07752)](https://arxiv.org/abs/2412.07752), [MPK mega-kernel compiler (OSDI 2026, arXiv 2512.22219)](https://arxiv.org/abs/2512.22219) for the per-SM task-graph idea (inference only; the tool does not apply).

#### F76 Programmatic Dependent Launch for the exact kernel chain

- What: launch each dependent custom kernel with `cudaLaunchKernelEx` and `cudaLaunchAttributeProgrammaticStreamSerialization`; the primary kernel calls `cudaTriggerProgrammaticLaunchCompletion`, the secondary runs its independent prologue and then `cudaGridDependencySynchronize`. Inter-kernel gaps shrink because the next grid is resident before the previous one drains.
- Why here: the accepted path chains shift gather, row-tile recurrence, GroupNorm and index kernels thousands of times per update, so launch-to-launch gaps are a direct component of the lockstep host/GPU time.
- Exactness: exact; only scheduling changes.
- First step: a two-kernel synthetic on sm_120 (support is documented for compute capability 9.0 and newer, Hopper and Blackwell per CUTLASS; GeForce is not singled out either way) measured with CUDA events, then the ctypes driver transport that already exists for exact launches gains the attribute. PDL edges also exist for CUDA graphs (`cudaGraphDependencyTypeProgrammatic`), so it composes with F05.
- Risks: the earlier direct ctypes transport was slower per call (4.9 → 13.2–13.6 µs); PDL gains must exceed that or ride on the native dispatcher.
- Sources: [CUDA Programming Guide: Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html), [CUTLASS dependent kernel launches](https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/dependent_kernel_launch.html).

#### F77 WDDM software-queue flush policy

- What: on WDDM the CUDA driver batches launches in a user-mode software queue before submitting a command buffer; `cudaStreamQuery`/`cudaEventQuery` (and any synchronous call) flush it. Test an explicit flush after each layer's launch burst, after each split, and none, against the accepted path.
- Why here: the measured host/GPU lockstep (summed dispatch ≈ summed GPU span) is what command-buffer back-pressure looks like, and small updates are host-bound. The repository already removed most synchronizing reads (F33, F37), so flush placement is the remaining knob.
- Exactness: exact.
- First step: count current implicit flushes per update with Nsight Systems' WDDM trace / CUDA driver queue latency rows (ordinary-rights capture failed earlier, so this may need the human); otherwise an A/B of flush placements inside the existing paired-timing harness. Per-launch flushing is expected to cost more than it saves.
- Risks: forum evidence is mixed and dated (Vista–Windows 8 era); NVIDIA moderators advise against querying after every launch.
- Sources: NVIDIA forum threads [Very slow kernel launches](https://forums.developer.nvidia.com/t/very-slow-kernel-launches/37345), [Is WDDM causing this?](https://forums.developer.nvidia.com/t/is-wddm-causing-this/29903), [Performance of launching kernels](https://forums.developer.nvidia.com/t/performance-of-lauching-kernels/32844).

#### F78 Native Linux comparison of the per-update floor

- What: boot the same machine into native Linux (not WSL2) with the same PyTorch/CUDA versions and measure the accepted path's small-update floor, launch cost and full-run throughput.
- Why here: Linux avoids the WDDM path entirely; one controlled measurement found small matrix products about 5.5× faster on Linux than Windows while large ones matched, and NVIDIA reports WSL2 at 90% or less of native speed on launch-heavy workloads. Hardware-accelerated GPU scheduling (F68) is documented only as scheduling between applications, with no claim about CUDA launch cost. This bounds every launch-side optimization on Windows (F05, F65, F75–F77): if the floor halves on Linux, that is the cheapest "optimization" available.
- Exactness: exact (same code); timing comparison only, with checkpoints and data read from the same disk.
- First step: human decision on a dual boot or a live USB with the NVIDIA driver; then the existing `check_r01_exact_options.py` and one paired full run.
- Sources: [Janea Systems: small-matrix matmul on Windows vs Linux](https://janeasystems.com/blog/pytorch-matrix-multiplication-cuda-bug), [NVIDIA: CUDA on WSL2 performance](https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements), [Microsoft: Hardware Accelerated GPU Scheduling](https://devblogs.microsoft.com/directx/hardware-accelerated-gpu-scheduling/).

#### F79 Triton on Windows as a second kernel path (Inductor, FLA and wind_rwkv kernels)

- What: install `triton-windows` 3.5 (the version paired with PyTorch 2.9; Blackwell sm_120 needs Triton ≥ 3.3, PyTorch ≥ 2.7, CUDA ≥ 12.8; the wheels bundle a CUDA 12.8 toolchain) in a separate venv, then (a) compile the time mixer with `torch.compile` to see what Inductor fuses, (b) benchmark `fla.ops.rwkv7` (`chunk.py`, `fused_recurrent.py`, `fused_addcmul.py`, `fused_k_update.py`) and wind_rwkv's Triton "bighead" kernel on real R01 shapes.
- Why here: it is the only way to get fused elementwise kernels without hand-writing CUDA, and it gives F69 a reference chunked implementation to measure against.
- Exactness: non-exact for Inductor (fusion changes order); Triton kernels written with explicit accumulation order can be exact, as the NVRTC kernels are.
- First step: environment smoke (`torch.cuda.get_arch_list()` must list sm_120, one Triton kernel runs), then a forward-only microbenchmark of the accepted row-tile kernel against `fused_recurrent_rwkv7` and `chunk_rwkv7` at H = 1, d = 32, real T distribution. BlinkDL notes that kernel speed scales with Bsz × HeadCount and that the FLA RWKV-7 implementation is "not aligned with the reference and performs worse", so expect the chunked kernels to need batch rows, not heads, for parallelism.
- Sources: [triton-windows](https://github.com/woct0rdho/triton-windows), [flash-linear-attention `fla/ops/rwkv7`](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/rwkv7), [wind_rwkv](https://github.com/johanwind/wind_rwkv), [RWKV-LM README](https://github.com/BlinkDL/RWKV-LM).

#### F80 CUDA caching-allocator configuration

- What: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the `cudaMallocAsync` backend as option-only variants; record allocator calls per update.
- Why here: every update copies batches into fresh device storage (`bulk_to_device`) and runs about 4,100 `fill_` calls; allocator retries/OOMs are already zero, so the expected gain is small and the track is Low priority. It costs one option and no code.
- Exactness: exact.
- First step: count `cudaMalloc`/`cudaFree` and segment growth per update from `torch.cuda.memory_stats()` in one accepted-path run; only if non-trivial, run the two variants in interleaved blocks.
- Sources: [PyTorch CUDA memory management notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management).

#### F81 RWKV-7 training-recipe audit (non-exact)

- What: compare `rwkv/train_rwkv.py`'s optimizer groups with the published RWKV-7 recipe: weight decay only on 2-D projection matrices (the README calls this "VERY IMPORTANT"), AdamW `eps` 1e-18, 2× learning rate for `w0`, LayerNorm weights decay-free, PreLN LayerNorm rather than RMSNorm, fp32 state update because `w` is close to 1, and the small-data advice (dropout at about a quarter of the usual value, deep-narrow shapes).
- Why here: the vendored trainer already uses five parameter groups and a 2-D decay rule; the audit checks which of the remaining settings differ and whether any matters for a 192,800-parameter model trained on review logs.
- Exactness: non-exact; quality candidate under the seed gate with unchanged exposure.
- First step: a read-only table (setting, upstream value, repository value) in this file; experiments only for settings that differ.
- Sources: [RWKV-LM README](https://github.com/BlinkDL/RWKV-LM), [RWKV-7 "Goose" paper (arXiv 2503.14456)](https://arxiv.org/abs/2503.14456).

#### F82 Muon optimizer for the projection matrices (non-exact)

- What: replace AdamW for 2-D weight matrices with Muon (orthogonalized momentum via Newton–Schulz), keeping AdamW for vectors, embeddings and heads, with weight decay and the per-parameter update-scale adjustment the Moonlight work found necessary.
- Why here: reported gains are sample efficiency (small 30M–200M models reach the AdamW target loss with 48–52% of the compute; Moonlight reports about 2× compute efficiency), which in R01 terms is a quality gain at fixed exposure. The matrices here are 32×32 (and 4×32-wide hidden layers), so the Newton–Schulz cost is negligible. Evidence is from the optimizer authors' groups and from language modelling; nothing on review-log models.
- Exactness: non-exact; seed gate on held-out Curve and Instant losses with unchanged exposure and update count; the throughput effect is expected to be neutral.
- Sources: [Practical efficiency of Muon for pretraining (arXiv 2505.02222)](https://arxiv.org/abs/2505.02222), [Muon is scalable / Moonlight (arXiv 2502.16982)](https://arxiv.org/abs/2502.16982), [arXiv 2509.24406](https://arxiv.org/abs/2509.24406).

#### F83 State structure variants from the RWKV-8 notes (non-exact)

- What: three directions from Bo Peng's "Improving RNNs" notes mapped to this model: larger state in only the slowest-decaying head (we have one 32-wide head per stream, so this means a second, wider head in the global stream), smaller state by sharing state across layers or low-rank/quantized state (the Anki deployment target is a per-card state near 1 KiB; iter35 cut card layers 2 → 1 for that reason), and mixed state, where the last layer's state of one review feeds the first layer of the next (claimed to make a depth-L model behave like depth 2L, with a backward-pass cost the author suggests paying only periodically or at readout).
- Why here: the five-stream model already trades layers between streams for state size (iter35/iter36 comments in `expertium/architecture.py`); these are the published next moves on that trade-off.
- Exactness: non-exact; quality and state-size candidates under the seed gate; speed effect unknown.
- First step: cost the state-size change of each variant from the architecture file before any training; mixed state is the only one that may also reduce layers (speed).
- Sources: [RWKV-8.md "Improving RNNs"](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md), [RWKV architecture history](https://wiki.rwkv.com/basic/architecture.html).

#### F84 Recency-weighted training loss (non-exact)

- What: weight each review's loss by its recency inside the user's history, as FSRS-7's recency weighting does, and judge on the same held-out Curve/Instant losses.
- Why here: in the upstream benchmark recency weighting improves FSRS-7 from 0.3401 to 0.3370 log loss; it is a one-line loss change with unchanged exposure and update count. Whether a sequence model that already sees the whole history benefits is unknown.
- Exactness: non-exact; seed gate.
- Sources: [srs-benchmark README](https://github.com/open-spaced-repetition/srs-benchmark).

#### F85 Fill small updates to the 65,536-review cap (non-exact: changes the update schedule)

- What: compose updates so that the 72 small updates below about 49k reviews (19.4% of reviews, about 35% of summed GPU span) are merged toward the 65,536 cap, reducing the update count.
- Why here: the per-update floor is nearly flat, so fewer, fuller updates convert directly into less GPU span (upper bound near 20% of training time). The cost is a changed update schedule, which the ledger rules require to be disclosed and which the seed gate must judge for quality.
- Exactness: non-exact (training behaviour changes even though per-update arithmetic does not).
- First step: an offline count of how many updates a cap-filling policy produces from the recorded batch sizes, then a seed-gated run.
- Sources: repository measurements only ([0070 cost split](R01_HOST_PATH.md#idea-scouting-after-0070-analysis-only-2026-10-13)).

### Notes for existing tracks (no new ID)

- F05 CUDA graphs: NVIDIA's PyTorch integration uses stream capture only, exposes no graph-update API and recaptures on parameter change; the common dynamic-size pattern is one graph per bucket with all graphs sharing one memory pool (PyTorch cudagraph trees), and lengths passed through device memory so a captured graph serves any length up to its bucket. CUDA 12.4 conditional nodes (if/else, loops) remove the need to recapture for control flow but still fix tensor shapes, and PyTorch does not expose them, so they need a C++/cuda-python extension. PDL edges inside graphs (F76) are available. Sources: [NVIDIA CUDA-graph constraints](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html), [PyTorch CUDA graph integration](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html), [best practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html).
- F69 chunked WKV7: wind_rwkv documents the kernel families and their error against the reference recurrence: chunked bf16 (chunk length 16, tensor-core matmuls) about 5e-3, backstepping fp32 (the official training kernel; steps the state backward up to 15 steps and limits decay to about 0.5) about 9e-5, Triton bighead fp32 about 5e-5, FLA about 4e-3. On an H100 the chunked bf16 kernel is 8–16× the backstepping kernel at head size 256, but all of these parallelize over Bsz × HeadCount; with one 32-wide head per stream the only parallelism is over rows in a split and along T, which argues for a chunk-level scan over T (the upstream `rwkv/model/csrc/cuda/parallel_scan.cu` and `rwkv7_cuda_time_parallel_*` headers already exist in this repository and should be the first thing to measure). Any chunked variant is non-exact and needs the seed gate; F22's +0.84–1.48% single-seed shift is the warning.
- F66 stall-guided backward: wind_rwkv's backstepping kernel is the published reference for the backward structure; the dense-shape Nsight counters remain the missing evidence.
- F68 WDDM environment: no source documents an effect of hardware-accelerated GPU scheduling on CUDA launch cost; the WDDM software queue is the documented mechanism (F77) and native Linux is the documented escape (F78).
- F72 main-thread budget: the Windows small-op penalty measured by Janea Systems (about 5.5× on small matmuls) supports measuring the floor on Linux before more host-side work.
- Upstream data path: srs-benchmark now opens LMDB per worker and raises the reader limit (Feb 2026) and makes process-priority setting portable; both are worth a diff against the vendored `rwkv/data_processing.py` the next time preparation workers are touched (F71/F73).

### Surveyed and not adopted

- RWKV-8 ROSA (Rapid Online Suffix Automaton, October 2025): a parameter-free copy mechanism for token sequences; nothing in review logs repeats verbatim the way text does.
- RWKV-8 DeepEmbed / DeepEmbedAttention (May 2025): learnable per-token channel scaling stored off-GPU; the natural analogue (per-card/deck/preset vectors) does not transfer to unseen users, which is how RWKV is evaluated.
- RWKV-7 "State Tuning" (arXiv 2504.05097): test-time state optimization; the model is not tuned per user by design.
- Mamba4KT / DGMKT (knowledge tracing with Mamba, 2024–2025): same linear-recurrence family as RWKV-7 with no evidence of a gain over it; Kalman linear attention (arXiv 2602.10743) is a language-modelling proposal without a student-modelling result.
- LECTOR (LLM-based scheduling): a scheduler, not a memory model.
- MPK (Mirage Persistent Kernel): inference-only, multi-GPU mega-kernel compiler; the idea is carried by F75, the tool is not applicable.
- Task Spooler, GPU task-spooler forks, `gq`, `gpu-gate`: Linux daemons or cooperative locks for GPU queues; the same cooperative-lock approach is implemented in-repository as `scripts/r01_queue.py` (FIFO tickets, Windows change notifications instead of polling, stale-owner removal) because sessions here need claims beyond the GPU and no daemon.
- Expertium's `rwkv-anki-autoresearch`: the CPU-inference, quantized-in-Anki goal and the iteration naming (`reference_iter31`…`iter45`) match this repository's `expertium/architecture.py` history; its README lists no new architecture or speed technique beyond what is here, but its `AGENTS.md` roadmap was not readable and is worth a look by the human.
