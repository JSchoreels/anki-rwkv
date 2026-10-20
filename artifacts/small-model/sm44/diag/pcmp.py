import sys, pickle
sys.path.insert(0, sys.argv[1])
from scripts import check_state_vq_score_gpu as chk
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[2:4])
c = chk.compare(a, b, 1e-5)
print({h: (c[h]["d_exact_logloss"], c[h]["d_benchmark_logloss"], c[h]["max_abs_dp"], c[h]["n_dp_over_1e-4"]) for h in ("ahead", "instant")}, "ok", c["ok"])
