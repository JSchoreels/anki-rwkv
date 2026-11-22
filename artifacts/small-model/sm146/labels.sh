# Aggressive state tiers for Tiny (SM146): the scored labels (design.md "Labels"), sourced by cpu-proof.sh and launch-sm146.sh.
# NAME=variant; spec b0.T.b1.N.b2 = b0 for answers n < T, b1 for T <= n < N, b2 for n >= N (n = the stored state's answers);
# the 3-part spec b0.N.b1 = b0 below N, b1 from N. Card spec before "c", note spec before "n". Cards and notes alike in every label.
REF=w8+s8hx8                          # int8 cards and notes (the release-candidate format, reference of the Tiny cost)
AN16=w8+t4.16.8c4.16.8nhx8            # int4 below 16 answers, int8 from 16 (SM144 AN16, the paired baseline)
I3a=w8+t3.5.4.16.8c3.5.4.16.8nhx8     # int3 for 1-4 answers, int4 for 5-15, int8 from 16
I3b=w8+t3.9.4.16.8c3.9.4.16.8nhx8     # int3 for 1-8 answers, int4 for 9-15, int8 from 16
I2a=w8+t2.5.4.16.8c2.5.4.16.8nhx8     # int2 (ternary) for 1-4 answers, int4 for 5-15, int8 from 16
I2b=w8+t2.5.3.16.8c2.5.3.16.8nhx8     # int2 (ternary) for 1-4 answers, int3 for 5-15, int8 from 16
U8=w8+t8.5.8.128.8c8.5.8.128.8nhx8    # proof only: every tier int8 (must equal REF)
P4=w8+t4.5.4.16.8c4.5.4.16.8nhx8      # proof only: the 5-part spec with the young tier at int4 (must equal AN16)
GPU_LABELS="REF AN16 I3a I3b I2a I2b" # the 732 set (GPU)
LONG_LABELS="REF AN16 I3a I3b I2a I2b" # the long-card users (CPU): every label scored directly, with the per-review dump
SMOKE_LABELS="REF AN16 I3a I3b I2a I2b U8 P4"
EDGES=0,1,3,5,9,16,32,128,300,1000    # --prior-buckets (answers before the review); 5 and 9 are the young cut-offs, 16 the int8 switch
