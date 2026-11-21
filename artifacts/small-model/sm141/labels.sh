# Tiered card state by answer count (SM141): the scored labels (design.md "Schemes"), sourced by cpu-proof.sh and launch-sm141.sh.
# NAME=variant; spec b0.T.b1.N.b2 = b0 for answers n < T, b1 for T <= n < N, b2 for n >= N (n = the stored state's answers).
REF=w8+s8hx8                          # int8 cards and notes (the release-candidate format, reference)
A=w8+t4.128.8c8nhx8                   # cards int4 below 128 answers, int8 from 128; notes int8
AN=w8+t4.128.8c4.128.8nhx8            # cards and notes int4 below 128, int8 from 128
B4=w8+t3.5.4.128.8c3.5.4.128.8nhx8    # AN + int3 for 1-4 answers (cards and notes)
B8=w8+t3.9.4.128.8c3.9.4.128.8nhx8    # AN + int3 for 1-8 answers
D4=w8+t2.5.4.128.8c2.5.4.128.8nhx8    # AN + int2 (ternary codes) for 1-4 answers (secondary)
U8=w8+t8.5.8.128.8c8.5.8.128.8nhx8    # proof only: every tier int8 (must equal REF)
U4=w8+t4.5.4.128.4c8.5.8.128.8nhx8    # proof only: int4 cards / int8 notes in every tier (must equal C4)
C4=w8+s4c8nhx8                        # proof only: compact int4 cards (the stock code)
GPU_LABELS="REF A AN B4 B8 D4"        # the 732 set (GPU)
LONG_LABELS="REF AN B8 D4"            # the long-card users (CPU); A and B4 by precision dominance (design.md)
SMOKE_LABELS="REF A AN B4 B8 D4 U8 U4 C4"
EDGES=0,1,3,5,9,17,33,128,300,1000    # --prior-buckets (answers before the review)
