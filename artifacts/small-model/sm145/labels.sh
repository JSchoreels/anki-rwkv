# Int3 young tier on the early switch (SM145): the scored labels (design.md "Labels"), sourced by cpu-proof.sh and launch-sm145.sh.
# NAME=variant; spec b0.T.b1.N.b2 = b0 for answers n < T, b1 for T <= n < N, b2 for n >= N (n = the stored state's answers);
# the 3-part spec b0.N.b1 = b0 below N, b1 from N. Card spec before "c", note spec before "n".
REF=w8+s8hx8                          # int8 cards and notes (the release-candidate format, reference)
AN16=w8+t4.16.8c4.16.8nhx8            # cards and notes int4 below 16 answers, int8 from 16 (SM144 AN16, the paired baseline)
Y2N=w8+t3.3.4.16.8c3.3.4.16.8nhx8     # cards and notes int3 for 1-2 answers, int4 for 3-15, int8 from 16
Y4C=w8+t3.5.4.16.8c4.16.8nhx8         # cards int3 for 1-4 answers, int4 for 5-15, int8 from 16; notes as AN16 (int4 below 16, int8 from 16)
Y4N=w8+t3.5.4.16.8c3.5.4.16.8nhx8     # cards and notes int3 for 1-4 answers, int4 for 5-15, int8 from 16
U8=w8+t8.5.8.128.8c8.5.8.128.8nhx8    # proof only: every tier int8 (must equal REF)
U4=w8+t4.5.4.128.4c8.5.8.128.8nhx8    # proof only: int4 cards / int8 notes in every tier (must equal C4)
C4=w8+s4c8nhx8                        # proof only: compact int4 cards (the stock code)
P4=w8+t4.5.4.16.8c4.5.4.16.8nhx8      # proof only: the 5-part spec with the young tier at int4 (must equal AN16)
GPU_LABELS="REF AN16 Y2N Y4C Y4N"     # the 732 set (GPU)
LONG_LABELS="REF AN16 Y2N Y4C Y4N"    # the long-card users (CPU): every label scored directly, with the per-review dump
DR_LABELS="AN16 Y2N Y4C Y4N"          # per-review |dR| read-out vs REF (long-card users)
SMOKE_LABELS="REF AN16 Y2N Y4C Y4N U8 U4 C4 P4"
EDGES=0,1,3,5,9,16,32,128,300,1000    # --prior-buckets (answers before the review); 3 and 5 are the young cut-offs, 16 the int8 switch
