# Release-candidate re-check of the final state formats (SM164): the scored labels (design.md "Labels"), sourced by
# launch-sm164-gpu.sh. Plain labels of SM148 / SM152, unchanged code (branch cut from sm152).
REF=w8+s8hx8               # int8 card and note rows: the baseline of every budget
T24=w8+lr2.16.4q8x8        # rank 2 below 16 answers, rank 4 from 16, int8 factors: the six-layer Tiny's final format
F6=w8+lr2.16.4q6x8         # T24 with int6 factors: the Big and Small final format (user 11-24)
EDGES=0,1,3,5,9,16,32,128,300,1000                        # --prior-buckets, as SM148 / SM152 (rows equal to theirs)
