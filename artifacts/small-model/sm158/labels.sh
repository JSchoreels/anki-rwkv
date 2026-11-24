# Output-aware T24 on the 732 set (SM158): the labels (design.md "Labels"), sourced by cpu-proof.sh and launch-sm158-gpu.sh.
# Grammar of scripts/state_lowrank.LowRankCodec: SM148's lr<spec>q<f>x<s> plus SM157's suffix w<kind>; here kind = rk
# (receptance + equal-trace removal-key term), M from SM157_STATS = ../sm157/sm157-stats.json (fitted in SM157, not refit).
REF=w8+s8hx8               # int8 cards and notes (SM148's 732 rows; the reference of every cost)
Q2=w8+lr2q8x8              # rank 2 everywhere, int8 factors, int8 shifts, plain truncation (SM148's 732 rows)
T24=w8+lr2.16.4q8x8        # the shipped format: rank 2 below 16 answers, rank 4 from 16, plain (SM148's 732 rows)
T24RK=w8+lr2.16.4q8x8wrk   # T24 with output-aware truncation, M = receptance + removal key (scored here; the decision)
Q2RK=w8+lr2q8x8wrk         # Q2 with the same weighting (scored here unless the ETA rule drops it; information)
SMOKE_OLD="REF Q2 T24"     # CPU proof: equal to SM148's committed CPU smoke rows
SMOKE_NEW="Q2RK T24RK"     # CPU proof: equal to SM157's CPU probe rows for the 3 smoke users
EDGES=0,1,3,5,9,16,32,128,300,1000                        # --prior-buckets (answers before the review); 16 is T24's switch
