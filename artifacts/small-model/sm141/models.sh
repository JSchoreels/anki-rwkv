# Tiered card state by answer count (SM141): the six release-candidate models (from the release-candidate checks (SM139) models.sh, role rc only).
# TAG|KIND:ARCH|PATH|SHA256|ROLE|SIZE (sourced by launch-sm141.sh and cpu-proof.sh)
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
MODELS="
sm91s2-big-showtime|fork:|$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth|67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590|rc|big
sm99-big-inst-e2|fork:|$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth|c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768|rc|big
sm118-e26-dec2ep32k-t91s2|expertium:w64|$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth|b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6|rc|small
sm113-inst-e35st-st-dec2ep32k-t99|expertium:w64|$W/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth|e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736|rc|small
sm130-th-dec2ep32k-t91s2|expertium:w64-c1-d2p1u1|$W/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/$TRACK/master.pth|48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af|rc|tiny
sm130-io-dec2ep32k-t99|expertium:w64-c1-d2p1u1|$W/sm130/artifacts/small-model/sm130-io-dec2ep32k-p2/expertium/$TRACK/master.pth|23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2|rc|tiny
"
