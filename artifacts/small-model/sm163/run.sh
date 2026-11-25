#!/bin/bash
# SM163 clean speed re-measure: before = fork main fb9f59a18, after = fork main eda153ddb.
# Release binaries in bin/, synthetic data only. Configs alternate before/after (ABAB), order reversed on rep 2.
S=C:/Users/jonat/AppData/Local/Temp/claude/eefeeb6d2df69e2b/41a2947b-8da0-4ee6-a033-bc96eb42a697/scratchpad/sm163
OUT=$S/out; mkdir -p $OUT $S/work
cd C:/Users/jonat/workspace/wt/fork-lowrank-speed/rslib
configs=(before:compact_cards_int8 after:compact_cards_int8 before:compact_lowrank_int8 after:compact_lowrank_int8 after:compact_lowrank after:compact_lowrank_int6)
probe() {
  echo "cpu_load_percent=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average" 2>/dev/null | tr -d '\r')"
  echo "gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d '\r')"
}
for rep in 1 2 3; do
  if [ $rep = 2 ]; then order=(5 4 3 2 1 0); else order=(0 1 2 3 4 5); fi
  for k in "${order[@]}"; do
    item=${configs[$k]}; bin=${item%%:*}; p=${item#*:}
    tag="rep=$rep binary=$bin precision=$p"
    echo "=== $(date '+%F %T') $tag"; probe
    for rt in big-pair small-pair tiny-pair; do
      SM163_RUNTIME=$rt SM163_PRECISION=$p SM163_WORK=$S/work SM163_TAG="$rep/$bin" SM163_OUT=$OUT/app.jsonl \
        $S/bin/sm163-$bin.exe 2>&1 | grep -E '^\{|skip|panick|error' | cut -c1-200
    done
    echo "--- one-thread 1000x24 $tag"; probe
    RAYON_NUM_THREADS=1 ANKI_RWKV_SIZE_BENCH_RUNTIMES=big,small,tiny ANKI_RWKV_SIZE_BENCH_SAMPLES=3 \
      ANKI_RWKV_SIZE_BENCH_CARDS=1000 ANKI_RWKV_SIZE_BENCH_ROUNDS=24 ANKI_RWKV_SIZE_BENCH_PRECISION=$p \
      $S/bin/test-$bin.exe --ignored --nocapture --exact rwkv::model_size_bench::rwkv_model_size_benchmark 2>&1 | grep -E "^(big|small|tiny|answers|rayon|thread)"
    echo "--- one-thread 4000x8 $tag"; probe
    RAYON_NUM_THREADS=1 ANKI_RWKV_SIZE_BENCH_RUNTIMES=small,tiny ANKI_RWKV_SIZE_BENCH_SAMPLES=3 \
      ANKI_RWKV_SIZE_BENCH_CARDS=4000 ANKI_RWKV_SIZE_BENCH_ROUNDS=8 ANKI_RWKV_SIZE_BENCH_PRECISION=$p \
      $S/bin/test-$bin.exe --ignored --nocapture --exact rwkv::model_size_bench::rwkv_model_size_benchmark 2>&1 | grep -E "^(big|small|tiny|answers|rayon|thread)"
    echo "--- four-thread 4000x8 $tag"; probe
    RAYON_NUM_THREADS=4 ANKI_RWKV_SIZE_BENCH_RUNTIMES=big,big-pair,small,small-pair,tiny,tiny-pair ANKI_RWKV_SIZE_BENCH_SAMPLES=3 \
      ANKI_RWKV_SIZE_BENCH_PRECISION=$p \
      $S/bin/test-$bin.exe --ignored --nocapture --exact rwkv::model_size_bench::rwkv_model_size_benchmark 2>&1 | grep -E "^(big|small|tiny|answers|rayon|thread)"
  done
done
echo "=== done $(date '+%F %T')"; probe
