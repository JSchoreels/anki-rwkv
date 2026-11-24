#!/bin/bash
# SM156 speed, second pass: Small and Tiny only (shorter runs), 5 rotations of the four configurations
# (frozen / current binary x int8 rows / T24), single thread, release build.
S=C:/Users/jonat/AppData/Local/Temp/claude/eefeeb6d2df69e2b/41a2947b-8da0-4ee6-a033-bc96eb42a697/scratchpad
cd C:/Users/jonat/workspace/wt/fork-lowrank-speed/rslib
export RAYON_NUM_THREADS=1 ANKI_RWKV_SIZE_BENCH_RUNTIMES=${RUNTIMES:-small,tiny} ANKI_RWKV_SIZE_BENCH_SAMPLES=3
configs=(old:compact_lowrank_int8 new:compact_lowrank_int8 old:compact_cards_int8 new:compact_cards_int8)
for shape in ${SHAPES:-"1000:24" "4000:8"}; do
  cards=${shape%%:*}; rounds=${shape#*:}
  for rep in $(seq 1 ${REPS:-5}); do
    for k in 0 1 2 3; do
      item=${configs[$(( (k + rep) % 4 ))]}
      bin=${item%%:*}; p=${item#*:}
      echo "=== $(date '+%F %T') cards=$cards rounds=$rounds rep=$rep binary=$bin precision=$p"
      powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).LoadPercentage" 2>/dev/null | tr -d '\r' | sed 's/^/cpu_load_percent=/'
      ANKI_RWKV_SIZE_BENCH_CARDS=$cards ANKI_RWKV_SIZE_BENCH_ROUNDS=$rounds ANKI_RWKV_SIZE_BENCH_PRECISION=$p \
        $S/bin/anki-$bin-default.exe --ignored --nocapture --exact rwkv::model_size_bench::rwkv_model_size_benchmark 2>&1 | grep -E "^(big|small|tiny)"
    done
  done
done
echo "=== done $(date '+%F %T')"
