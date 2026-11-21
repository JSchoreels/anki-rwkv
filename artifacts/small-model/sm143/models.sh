# Per-user adaptation (SM143): the model table (sourced by launch-sm143.sh and cpu-proof.sh).
# TAG|KIND:ARCH|PATH|SHA256|ROLE (adapt = adapted with variants a,b,c (+ d for the primary), ref = unadapted reference)
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
SMALL="sm113-inst-e35st-st-dec2ep32k-t99|expertium:w64|$W/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth|e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736|adapt"
TINY="sm130-io-dec2ep32k-t99|expertium:w64-c1-d2p1u1|$W/sm130/artifacts/small-model/sm130-io-dec2ep32k-p2/expertium/$TRACK/master.pth|23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2|adapt"
BIG="sm99-big-inst-e2|fork:|$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth|c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768|ref"
FSRS=$W/sm81/artifacts/small-model/sm81/gpu/fsrs
