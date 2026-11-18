# SM32c: learned codebooks (~400 bits/state) in the one-epoch 32k decay of SM42, 68 users, w8 weights

Review-weighted LogLoss, paired user bootstrap (2,000 resamples, seed 0, 95% CI); diff = first - second; ahead = curve.

```
{'sm42/w8+vq378x412': 68, 'sm42/w8+s8': 68, 'sm32c/w8+vq378x412': 68, 'sm32c/w8+s8': 68}
sm32c/w8+vq378x412 - sm42/w8+vq378x412 (68 users)
  ahead   0.31191 vs 0.31205 diff -0.00014 [-0.00080, +0.00039] -0.05%
  instant 0.28269 vs 0.28308 diff -0.00039 [-0.00065, -0.00016] -0.14%
sm32c/w8+vq378x412 - sm42/w8+s8 (68 users)
  ahead   0.31191 vs 0.31073 diff +0.00118 [+0.00055, +0.00173] +0.38%
  instant 0.28269 vs 0.28049 diff +0.00221 [+0.00173, +0.00273] +0.79%
sm32c/w8+s8 - sm42/w8+s8 (68 users)
  ahead   0.31378 vs 0.31073 diff +0.00305 [+0.00201, +0.00416] +0.98%
  instant 0.28414 vs 0.28049 diff +0.00365 [+0.00284, +0.00459] +1.30%
sm42/w8+vq378x412 - sm42/w8+s8 (68 users)
  ahead   0.31205 vs 0.31073 diff +0.00132 [+0.00044, +0.00211] +0.42%
  instant 0.28308 vs 0.28049 diff +0.00260 [+0.00202, +0.00325] +0.93%
sm32c/w8+vq378x412 - sm32c/w8+s8 (68 users)
  ahead   0.31191 vs 0.31378 diff -0.00187 [-0.00266, -0.00113] -0.60%
  instant 0.28269 vs 0.28414 diff -0.00145 [-0.00197, -0.00102] -0.51%
```

Verdict: learning the codebooks lowers the compression cost only a little (+0.42% / +0.93% -> +0.38% / +0.79% vs SM42 with int8 state); the model trained this way is worse with int8 state (+0.98% / +1.30%).
