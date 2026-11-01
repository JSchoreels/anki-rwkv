#!/bin/bash
# Codebook state compression sweep (SM90): after the 900k WKV fit, assemble codebooks-big.npz (900k WKV + the same
# shift books, 6/8-bit norms hi = max), pick 200k vs 900k (choose_books.py), install the winner as codebooks-sm90.npz,
# pin its sha256 and re-run the book-dependent CPU proof (codec proof + CPU check references).
WT=C:/Users/jonat/workspace/wt/sm90
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
H=$WT/artifacts/small-model/sm90
L=$H/logs
export OMP_NUM_THREADS=4 VQ_FIT_THREADS=4
cd $WT
until grep -q '\[fitbig\] done' $L/fitbig.nohup 2>/dev/null; do sleep 15; done
A="--out-dir artifacts/small-model/sm90/books"
for f in fitbig-c0 fitbig-c1 fitbig-c2 fitbig-c35 fitbig-c467; do [ -f $H/books/$f.npz ] || { echo "[finalize2] $f missing (timeout?): keeping the 200k books"; exit 0; }; done
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A merge --files fitbig-c0.npz fitbig-c1.npz fitbig-c2.npz fitbig-c35.npz fitbig-c467.npz \
  fit-shift-t.npz fit-shift-c.npz --out codebooks-big-merged.npz > $L/finalize2.log 2>&1
"$PY" -X utf8 $WT/scripts/state_vq_fit.py $A refit-norms --codebooks codebooks-big-merged.npz --holdout-every 10 \
  --max-points 200000 --norm-bits 6 8 --norm-top max --out codebooks-big.npz >> $L/finalize2.log 2>&1
[ -f $H/books/codebooks-sm90-200k.npz ] || cp $H/books/codebooks-sm90.npz $H/books/codebooks-sm90-200k.npz
WIN=$("$PY" -X utf8 $H/choose_books.py codebooks-sm90-200k.npz codebooks-big.npz 2>> $L/finalize2.log | tail -n 1)
echo "[finalize2] winner $WIN"
cp $H/books/$WIN $H/books/codebooks-sm90.npz
"$PY" -X utf8 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" $H/books/codebooks-sm90.npz > $H/books/codebooks-sm90.sha256
echo "[finalize2] books $(cat $H/books/codebooks-sm90.sha256) $(date +%H:%M:%S)"
LOGC0=fit-c0.log; [ "$WIN" = codebooks-big.npz ] && LOGC0=fitbig-c0.log
"$PY" -X utf8 $H/proof_codec.py codebooks-sm90.npz $LOGC0 > $L/proof-codec.log 2>&1; echo "[finalize2] codec proof exit $?"
for part in a b c; do
  "$PY" -X utf8 $WT/scripts/check_state_vq_score_gpu.py cpu --cases $H/check-cases-$part.json --users 2 \
    --output $H/check-sm90-cpu-$part.json --ref-dir $H/check-ref > $L/check-cpu-$part.log 2>&1 &
done
wait
grep -h 'ALL OK\|FAIL' $L/check-cpu-?.log
echo "[finalize2] done $(date +%H:%M:%S)"
