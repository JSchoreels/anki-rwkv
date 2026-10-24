#!/bin/bash
# SM70: CPU per-review dump of the SM56 best checkpoint (w8+s8) on the first 200 users of the 458-user SM56 list,
# 5 processes x 2 threads, 40 users each (the list order is stratified over history-length deciles).
cd /c/Users/jonat/workspace/wt/sm70
P=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
CK=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/scale-w64kd-final-sc-lanes-dec10-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
O=artifacts/small-model/sm70
for i in 0 1 2 3 4; do
  $P scripts/small_model_heldout.py --model sm56=expertium:w64:$CK --variants w8+s8 --device cpu \
    --users-file artifacts/small-model/sm56/eval-users-464-fit.json --start $((i*40)) --count 40 --time-limit 20000 \
    --out $O/heldout-sm70-part$i.jsonl --dump $O/dump > $O/dump-part$i.log 2>&1 &
done
wait
echo "[sm70 dump] done"
