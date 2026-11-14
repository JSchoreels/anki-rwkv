#!/usr/bin/env bash
# Scheduling-time vs review-time accuracy (SM81): CPU proof of the scorer's --start-of-day option on 3 small users of the
# 732 set (2388, 4676, 1712; 3.7-4.1k reviews), leak-free inputs, for the released small model (the leak-free small
# model (SM72s), cool-down from epoch 25) and the big model (1 leak-free epoch), variants w8+s8 and fp32.
#   orig = the scorer before the change (git show of the branch point), off = the patched scorer without the option,
#   sod  = the patched scorer with --start-of-day --dump. scripts/sm81_check.py compares them.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm81
O=$WT/artifacts/small-model/sm81/cpu-proof
mkdir -p "$O"; rm -rf "$O"/*.jsonl "$O"/dump-*
echo '{"users": [2388, 4676, 1712]}' > "$O/users3.json"
BASE=d7726fd8   # sm81's branch point (origin/sm77)
git -C "$WT" show $BASE:scripts/small_model_heldout.py > "$WT/scripts/_sm81_orig_heldout.py"
SMALL=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
rc=0
for m in "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon=expertium:w64:$SMALL" "big-leakfree-1ep=fork::$BIGCK"; do
  tag=${m%%=*}
  for v in w8+s8 fp32; do
    common="--device cpu --pool 732 --users-file $O/users3.json --model $m --only $tag/$v --split-existing --deleted-cards neighbour"
    n=$tag-$v
    env -u RWKV_DELETED_CARDS "$PY" -X utf8 "$WT/scripts/_sm81_orig_heldout.py" $common --out "$O/orig-$n.jsonl" > "$O/run-$n.log" 2>&1 || rc=1
    env -u RWKV_DELETED_CARDS "$PY" -X utf8 "$WT/scripts/small_model_heldout.py" $common --out "$O/off-$n.jsonl" >> "$O/run-$n.log" 2>&1 || rc=1
    env -u RWKV_DELETED_CARDS "$PY" -X utf8 "$WT/scripts/small_model_heldout.py" $common --start-of-day --dump "$O/dump-$n" \
      --out "$O/sod-$n.jsonl" >> "$O/run-$n.log" 2>&1 || rc=1
    tol=1e-6; [ "$v" = fp32 ] || tol=1e-3
    "$PY" -X utf8 "$WT/scripts/sm81_check.py" compare --orig "$O/orig-$n.jsonl" --off "$O/off-$n.jsonl" --sod "$O/sod-$n.jsonl" \
      --dump "$O/dump-$n" --tol $tol --out "$O/check-$n.json" | tee -a "$O/cpu-proof.log" || rc=1
  done
done
rm -f "$WT/scripts/_sm81_orig_heldout.py"
echo "[sm81-proof] exit $rc" | tee -a "$O/cpu-proof.log"
exit $rc
