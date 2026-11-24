#!/bin/bash
# SM151: Deck Options replay savings at 4 threads, int8 rows vs low-rank, back to back.
cd C:/Users/jonat/workspace/wt/fork-lowrank
export PATH="/c/Users/jonat/.cargo/bin:$PATH" CARGO_TARGET_DIR=$PWD/out/rust RAYON_NUM_THREADS=4
export ANKI_RWKV_SIZE_BENCH_RUNTIMES=big,big-pair,big-instant,small,small-pair,small-instant,tiny,tiny-pair,tiny-instant ANKI_RWKV_SIZE_BENCH_SAMPLES=3
for rep in 1 2; do
  for p in compact_cards_int8 compact_lowrank_int8; do
    echo "=== $(date '+%F %T') threads=4 rep=$rep precision=$p"
    C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status 2>&1 | sed -n 2p
    powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).LoadPercentage" 2>/dev/null | tr -d '\r' | sed 's/^/cpu_load_percent=/'
    ANKI_RWKV_SIZE_BENCH_PRECISION=$p cargo test --release -q -p anki --lib rwkv_model_size_benchmark -- --ignored --nocapture 2>&1 | grep -E "^(big|small|tiny|answers|rayon)"
  done
done
echo "=== done $(date '+%F %T')"
