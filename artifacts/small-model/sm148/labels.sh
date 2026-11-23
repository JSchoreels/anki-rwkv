# Low-rank int8 card and note states, post-training (SM148): the scored labels (design.md "Labels"), sourced by
# cpu-proof.sh and launch-sm148.sh. SM148 variant lr<spec><q>x8 (scripts/state_lowrank.LowRankCodec): spec kind[.N.kind],
# kind = rank r (rank min(n, r) for an entity with n answers) or R (REF's int8 rows) for answer counts from N on;
# q8 = int8 factor vectors (symmetric, one float16 scale per vector), f = float32 factors; x8 = REF's int8 shift vectors.
REF=w8+s8hx8               # int8 cards and notes (the release-candidate format, reference of every rule)
Q4=w8+lr4q8x8              # rank 4, int8 factors, every state
T24=w8+lr2.16.4q8x8        # rank 2 below 16 answers, rank 4 from 16, int8 factors
Q2=w8+lr2q8x8              # rank 2, int8 factors, every state
U8=w8+lrRq8x8              # proof only: the SM148 code path with every write in REF's int8 rows (must equal REF)
F4=w8+lr4fx8               # proof only: rank 4, float32 factors, int8 shifts (the truncation without factor rounding)
F4X0=w8+lr4fx0             # proof only: rank 4, float32 factors and shifts = SM147's lr4 through the factored solve
LR4=w8+lr4                 # proof only: SM147's rank-4 truncation (float64 SVD, float32 shifts; CPU only)
GPU_LABELS="REF Q4 T24 Q2"                 # the 732 set (GPU)
LONG_LABELS="REF Q4 T24 Q2"                # the long-card users (CPU): every label scored directly, with the per-review dump
DR_LABELS="Q4 T24 Q2"                      # per-review |dR| read-out vs REF (long-card users)
SMOKE_LABELS="REF Q4 T24 Q2 U8 F4 F4X0 LR4"   # CPU proofs (LR4 CPU only)
GPU_SMOKE_LABELS="REF Q4 T24 Q2 U8 F4"
EDGES=0,1,3,5,9,16,32,128,300,1000         # --prior-buckets (answers before the review); 16 is T24's switch
