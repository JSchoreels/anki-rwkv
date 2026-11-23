"""SM148 coverage helper: run scripts/small_model_heldout.py unchanged (runpy) with a sampler thread that logs the working
set and the main thread's innermost frames every 2 s and exits the process above LIMIT GB. Usage: memprobe.py LOG LIMIT_GB -- scorer args"""
import os, runpy, sys, threading, time, traceback
import psutil
log, limit = sys.argv[1], float(sys.argv[2]); args = sys.argv[4:]
main_id = threading.main_thread().ident
def sampler():
    p = psutil.Process()
    with open(log, "a") as f:
        while True:
            ws = p.memory_info().rss / 2**30
            fr = sys._current_frames().get(main_id)
            st = traceback.extract_stack(fr)[-3:] if fr else []
            f.write(f"{time.strftime('%H:%M:%S')} {ws:.2f} " + " | ".join(f"{os.path.basename(s.filename)}:{s.lineno}:{s.name}" for s in st) + "\n"); f.flush()
            if ws > limit:
                f.write("ABORT over limit\n"); f.flush(); os._exit(99)
            time.sleep(2)
threading.Thread(target=sampler, daemon=True).start()
sys.argv = ["small_model_heldout.py"] + args
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "scripts")]
runpy.run_path("scripts/small_model_heldout.py", run_name="__main__")
