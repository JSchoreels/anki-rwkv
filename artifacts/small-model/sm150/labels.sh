# Rank-1 int8 states (SM150): the scored labels (design.md "Labels"), sourced by cpu-proof.sh and launch-sm150.sh.
# Variant lr<spec><q>x8 (scripts/state_lowrank.LowRankCodec, unchanged since SM148): spec kind[.N.kind], kind = rank r
# (rank min(n, r) for an entity with n answers) for answer counts from N on; q8 = int8 factor vectors, f = float32 factors;
# x8 = REF's int8 shift vectors, x0 = float32 shifts (proof only).
REF=w8+s8hx8              # int8 cards and notes (SM148's rows are reused for REF, Q4, Q2; REF rescored on the 3 smoke users only)
R1=w8+lr1q8x8             # rank 1 everywhere, int8 factors
R12=w8+lr1.16.2q8x8       # rank 1 below 16 answers, rank 2 from 16, int8 factors
F1=w8+lr1fx8              # proof only: rank 1, float32 factors, int8 shifts (GPU smoke vs CPU)
F1X0=w8+lr1fx0            # proof only: rank 1, float32 factors and shifts = SM147's lr1 through the factored solve
LR1=w8+lr1                # proof only: SM147's rank-1 truncation (float64 SVD, float32 shifts; CPU only)
GPU_LABELS="R12 R1"                        # the 732 set (GPU): new labels only
LONG_LABELS="R12 R1"                       # the long-card users (CPU): new labels only, with the per-review dump
DR_LABELS="R12 R1"                         # per-review |dR| read-out vs REF (REF dumps = SM148's, linked in at the aggregate step)
SMOKE_LABELS="REF R1 R12 F1 F1X0 LR1"      # CPU proofs (LR1 CPU only)
GPU_SMOKE_LABELS="REF R1 R12 F1"
EDGES=0,1,3,5,9,16,32,128,300,1000         # --prior-buckets (answers before the review); 16 is R12's switch
