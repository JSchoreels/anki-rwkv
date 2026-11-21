cd C:/Users/jonat/workspace/wt/sm142
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm142; X="$PY -X utf8 scripts/sm142_expr.py"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in sm130th sm130io sm113 sm118 sm99; do $X analyse --model $m > $D/gpu/analyse-$m.out 2>&1; echo "$m exit $?" >> $D/gpu/analyse-chain.log; done
echo chain-done >> $D/gpu/analyse-chain.log
