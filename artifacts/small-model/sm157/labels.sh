# Output-aware low-rank truncation (SM157): the scored labels (design.md "Labels"), sourced by cpu-proof.sh, run-sm157.sh
# and launch-sm157-gpu.sh. Variant lr<spec><q>x<s>[w<kind>] (scripts/state_lowrank.LowRankCodec): SM148's grammar plus
# the SM157 suffix w<kind> = output-aware truncation with M of kind r (receptance), rk (receptance + equal-trace
# removal-key term) or g (discounted observability Gramian); M from SM157_STATS (sm157_fit.py on training users).
REF=w8+s8hx8               # int8 cards and notes (the reference of every cost)
Q2=w8+lr2q8x8              # rank 2 everywhere, int8 factors, int8 shifts (plain)
T24=w8+lr2.16.4q8x8        # the shipped format: rank 2 below 16 answers, rank 4 from 16 (plain)
Q2R=w8+lr2q8x8wr
Q2RK=w8+lr2q8x8wrk
Q2G=w8+lr2q8x8wg
T24R=w8+lr2.16.4q8x8wr
T24RK=w8+lr2.16.4q8x8wrk
T24G=w8+lr2.16.4q8x8wg
PROBE_LABELS="REF Q2 T24 Q2R Q2RK Q2G T24R T24RK T24G"   # the 8 probe users (CPU)
SMOKE_LABELS="REF Q2 T24"                                 # bit identity vs SM148's CPU smoke rows (3 smoke users)
PIPE_LABELS="Q2R Q2RK T24G"                               # pipeline check, user 2388 only, test statistics (2 users)
EDGES=0,1,3,5,9,16,32,128,300,1000                        # --prior-buckets (answers before the review); 16 is T24's switch
