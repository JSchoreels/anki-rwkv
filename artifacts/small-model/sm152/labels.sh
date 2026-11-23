# Smaller shift vectors and factors on T24 states (SM152): the scored labels (design.md "Labels"), sourced by
# cpu-proof.sh and launch-sm152.sh. Variant lr<spec><q>x<s> (scripts/state_lowrank.LowRankCodec): spec kind[.N.kind],
# kind = rank r (rank min(n, r) for an entity with n answers) for answer counts from N on; q8 = int8 factor vectors,
# q6 = int6 factor vectors (SM152), one float16 scale per vector; x8 = REF's int8 shift vectors, x4 = int4 shift vectors
# (SM152, codes in [-7, 7], one float16 scale per vector).
REF=w8+s8hx8               # int8 cards and notes (SM148's rows are reused for REF and T24; rescored on the 3 smoke users only)
T24=w8+lr2.16.4q8x8        # the user's 11-23 choice: rank 2 below 16 answers, rank 4 from 16, int8 factors, int8 shifts
Q2=w8+lr2q8x8              # integrity only: rank 2, int8 factors (CPU smoke vs SM148's rows)
S4=w8+lr2.16.4q8x4         # T24 + int4 shift vectors
F6=w8+lr2.16.4q6x8         # T24 + int6 factors, int8 shift vectors
F6S4=w8+lr2.16.4q6x4       # T24 + int6 factors + int4 shift vectors
GPU_LABELS="S4 F6 F6S4"                    # the 732 set (GPU): new labels only
LONG_LABELS="S4 F6 F6S4"                   # later step (longcard mode, CPU, not started with the GPU run): new labels only
DR_LABELS="S4 F6 F6S4"                     # per-review |dR| read-out vs REF (REF dumps = SM148's), later step
SMOKE_LABELS="REF T24 Q2 S4 F6 F6S4"       # CPU proofs
GPU_SMOKE_LABELS="REF T24 S4 F6 F6S4"
EDGES=0,1,3,5,9,16,32,128,300,1000         # --prior-buckets (answers before the review); 16 is T24's switch
