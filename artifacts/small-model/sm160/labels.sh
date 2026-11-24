# Deploy-time ensemble of the two instant predictions (SM160): the scored labels (design.md "Labels"), sourced by
# launch-sm160-gpu.sh. Plain labels of the low-rank int8 states test (SM148), unchanged code.
T24=w8+lr2.16.4q8x8        # the shipped state format: rank 2 below 16 answers, rank 4 from 16, int8 factors (the decision)
REF=w8+s8hx8               # int8 card and note rows (information, scored after T24)
EDGES=0,1,3,5,9,16,32,128,300,1000                        # --prior-buckets, as SM148 / SM158 (rows equal to SM148's)
