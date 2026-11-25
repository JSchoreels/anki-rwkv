# Tiny ensemble on the 1,500 users (SM169): the two six-layer Tiny models (copy of the SM160 models.sh Tiny lines).
# TAG|KIND:ARCH|PATH|SHA256|ROLE|SIZE (sourced by launch-sm169-gpu.sh)
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
MODELS="
sm130-th-dec2ep32k-t91s2|expertium:w64-c1-d2p1u1|$W/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/$TRACK/master.pth|48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af|rc|tiny
sm130-io-dec2ep32k-t99|expertium:w64-c1-d2p1u1|$W/sm130/artifacts/small-model/sm130-io-dec2ep32k-p2/expertium/$TRACK/master.pth|23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2|rc|tiny
"
