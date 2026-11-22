# Earlier int8 switch (SM144): the scored labels (design.md "Schemes"), sourced by cpu-proof.sh and launch-sm144.sh.
# NAME=variant; spec b0.T.b1.N.b2 = b0 for answers n < T, b1 for T <= n < N, b2 for n >= N (n = the stored state's answers);
# the 3-part spec b0.N.b1 = b0 below N, b1 from N (used by every tiered label here).
REF=w8+s8hx8                          # int8 cards and notes (the release-candidate format, reference)
A=w8+t4.128.8c8nhx8                   # cards int4 below 128 answers, int8 from 128; notes int8 (SM141 scheme A, now measured directly)
A32=w8+t4.32.8c8nhx8                  # cards int4 below 32 answers, int8 from 32; notes int8
A16=w8+t4.16.8c8nhx8                  # cards int4 below 16 answers, int8 from 16; notes int8
AN32=w8+t4.32.8c4.32.8nhx8            # cards and notes int4 below 32, int8 from 32
AN16=w8+t4.16.8c4.16.8nhx8            # cards and notes int4 below 16, int8 from 16
U8=w8+t8.5.8.128.8c8.5.8.128.8nhx8    # proof only: every tier int8 (must equal REF)
U4=w8+t4.5.4.128.4c8.5.8.128.8nhx8    # proof only: int4 cards / int8 notes in every tier (must equal C4)
C4=w8+s4c8nhx8                        # proof only: compact int4 cards (the stock code)
GPU_LABELS="REF A A32 A16 AN32 AN16"  # the 732 set (GPU)
LONG_LABELS="REF A A32 A16 AN32 AN16" # the long-card users (CPU): every label scored directly, no dominance shortcut
SMOKE_LABELS="REF A A32 A16 AN32 AN16 U8 U4 C4"
EDGES=0,1,3,5,9,16,32,128,300,1000    # --prior-buckets (answers before the review); 16 and 32 are the switch points
