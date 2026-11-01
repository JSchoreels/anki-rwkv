#!/usr/bin/env bash
# 2026-10-17 overnight chain (base trained at >= 16k, docs/SMALL_MODEL.md SM22 open question):
# (1) cont: ctx8192-cont e8 -> e9 at 16384 x 2, 0.25-epoch decay at 16384, 464-user scoring of e9 and dec9
#     (dec9 vs dec8-L16k = paired effect of one more epoch at 16k);
# (2) only after "done: cont": 32768 x 1 decay from that e9 (dec9-L32k vs dec8-L32k).
SM=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model
TIER=gate FROM=8 TO=9 CHUNK=16384 RUN_TAG=ctx8192-cont SRC_LBL=s-w64kd-gate-ctx8192c FIRST_MIN=210 \
  bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-17-cont16k.log 2>&1
grep -q "done: cont" $SM/runbook-2026-10-17-cont16k.log || { echo "cont did not finish, no 32k decay"; exit 1; }
SRC_NAME=scale-w64kd-gate-cached-ctx16384-cont EPOCH=9 DCHUNK=32768 DBATCH=1 \
  bash $SM/runbook-2026-10-16d.sh longdecay > $SM/runbook-2026-10-17-longdecay32k.log 2>&1
