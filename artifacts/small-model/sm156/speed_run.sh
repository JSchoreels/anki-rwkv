#!/bin/bash
# SM156 speed: frozen (rc-2026-11 514f3a41d) vs current codec binaries, int8 rows and T24, back to back,
# single thread, release build, rwkv_model_size_benchmark.
S=C:/Users/jonat/AppData/Local/Temp/claude/eefeeb6d2df69e2b/41a2947b-8da0-4ee6-a033-bc96eb42a697/scratchpad
cd C:/Users/jonat/workspace/wt/fork-lowrank-speed/rslib
export RAYON_NUM_THREADS=1 ANKI_RWKV_SIZE_BENCH_RUNTIMES=big,small,tiny ANKI_RWKV_SIZE_BENCH_SAMPLES=3
for shape in "1000 24" "4000 8"; do
  set -- $shape
  for rep in 1 2; do
    if [ $rep = 1 ]; then order="old:compact_cards_int8 old:compact_lowrank_int8 new:compact_lowrank_int8 new:compact_cards_int8";
    else order="new:compact_cards_int8 new:compact_lowrank_int8 old:compact_lowrank_int8 old:compact_cards_int8"; fi
    for item in $order; do
      bin=${item%%:*}; p=${item#*:}
      echo "=== $(date '+%F %T') cards=$1 rounds=$2 rep=$rep binary=$bin precision=$p"
      C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status 2>&1 | sed -n 2p
      powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).LoadPercentage" 2>/dev/null | tr -d '\r' | sed 's/^/cpu_load_percent=/'
      ANKI_RWKV_SIZE_BENCH_CARDS=$1 ANKI_RWKV_SIZE_BENCH_ROUNDS=$2 ANKI_RWKV_SIZE_BENCH_PRECISION=$p \
        $S/bin/anki-$bin-default.exe rwkv_model_size_benchmark --ignored --nocapture --exact rwkv::model_size_bench::rwkv_model_size_benchmark 2>&1 | grep -E "^(big|small|tiny|answers|rayon)"
    done
  done
done
echo "=== done $(date '+%F %T')"
