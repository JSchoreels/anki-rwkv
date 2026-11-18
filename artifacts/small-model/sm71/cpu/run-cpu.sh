#!/bin/bash
# SM71 CPU estimate of view 1 (inputs without the deleted-card signals): SM56 and the big model (fork w8+s8) on the
# first 40 users of the SM56 458-user list (= SM70 part 0, stratified over history-length deciles), modes flags and
# neighbour, 4 processes x 2 threads, with the existing/deleted split. Baselines: SM70's CPU rows/dump (SM56) and the
# SM24 GPU dump (big model).
cd /c/Users/jonat/workspace/wt/sm71
P=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CK=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/scale-w64kd-final-sc-lanes-dec10-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
FK=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
O=artifacts/small-model/sm71/cpu
U=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/eval-users-464-fit.json
for mode in flags neighbour; do
  $P -X utf8 scripts/small_model_heldout.py --model sm56=expertium:w64:$CK --variants w8+s8 --device cpu --users-file $U \
    --count 40 --time-limit 20000 --deleted-cards $mode --split-existing --out $O/$mode-sm56.jsonl > $O/$mode-sm56.log 2>&1 &
  $P -X utf8 scripts/small_model_heldout.py --model fork=fork::$FK --variants w8+s8 --device cpu --users-file $U \
    --count 40 --time-limit 20000 --deleted-cards $mode --split-existing --out $O/$mode-fork.jsonl > $O/$mode-fork.log 2>&1 &
done
wait
echo "[sm71 cpu] done"
