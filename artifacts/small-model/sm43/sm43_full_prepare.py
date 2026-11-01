"""SM43 full (10-20): data diversity vs epochs at equal reviews seen, at full size, with an LR decay at the end.

  A: 1,250 gate training users x (10 constant-LR epochs + 1 decay epoch)
     (stratified by history length: 125 per length decile)
  B:   125 of those users     x (100 constant-LR epochs + 10 decay epochs)
     (12-13 per decile, picked so 10 x B's reviews per epoch = A's reviews per epoch within 0.5%)

Both: the 2,048-window cached-KD recipe of scale-w64kd-gate-cached-ctx2048 (same options, seed, order seed), from
scratch. Decay: scripts/lr_tail_decay.py (accuracy-loop epoch_runner hook), a linear LR ramp 7e-4 -> 0 over the last
N epochs (A: N = 1, B: N = 10), continuing from the constant-LR weights and AdamW state; KD uses decay_alpha (0.5)
during the decay epochs, as every lr_schedule run of the trainer does. Same reviews in the decay phase for both arms.

Run once (CPU): prepares two accuracy-loop experiments, then narrows their run.json user lists.
"""
import json
import random
import subprocess
from pathlib import Path

WT = Path("C:/Users/jonat/workspace/wt/small-model")
PY = "C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe"
EXP = WT / "artifacts/accuracy/experiments"
REF = json.loads((EXP / "scale-w64kd-gate-cached-ctx2048/experiment.json").read_text())
OUT = WT / "artifacts/small-model/sm43"
A_PER_DECILE, B_USERS, CONST_A, CONST_B, DECAY_A, DECAY_B = 125, 125, 10, 100, 1, 10

tiers = json.loads((WT / "artifacts/accuracy/tiers.json").read_text())
gate = tiers["tiers"]["gate"]["train_users"]
counts = json.loads((EXP / "scale-w64kd-gate-cached-ctx2048/run/run.json").read_text())["review_counts"]
rc = lambda u: counts[str(u)]
by_len = sorted(gate, key=lambda u: (rc(u), u))
deciles = [by_len[i * len(by_len) // 10:(i + 1) * len(by_len) // 10] for i in range(10)]
rng = random.Random(4343)
a_by_dec = [sorted(rng.sample(d, A_PER_DECILE)) for d in deciles]
users_a = sorted(u for d in a_by_dec for u in d)
ra = sum(map(rc, users_a))
best = None
for seed in range(5000):
    r = random.Random(seed)
    thirteen = set(r.sample(range(10), B_USERS - 12 * 10))  # 5 deciles get 13 users, 5 get 12
    b = sorted(u for i, d in enumerate(a_by_dec) for u in r.sample(d, 13 if i in thirteen else 12))
    err = abs(sum(map(rc, b)) * 10 / ra - 1)
    if best is None or err < best[0]:
        best = (err, seed, b, sorted(thirteen))
    if err < 0.005:
        break
err, seed_b, users_b, thirteen = best
assert err < 0.005, err
assert len(users_b) == B_USERS and len(users_a) == 10 * A_PER_DECILE and set(users_b) <= set(users_a)
rb = sum(map(rc, users_b))

overrides = REF["option_overrides"]
arms = {"sm43full-a-1250u-11ep": (users_a, CONST_A, DECAY_A, ra), "sm43full-b-125u-110ep": (users_b, CONST_B, DECAY_B, rb)}
for name, (users, const, decay, reviews) in arms.items():
    epochs = const + decay
    if not (EXP / name).exists():
        subprocess.run([PY, "-X", "utf8", str(WT / "scripts/accuracy_loop.py"), "prepare", name, "--tier", "gate",
                        "--seed", str(REF["seed"]), "--order-seed", str(REF["order_seed"]), "--epochs", str(epochs),
                        "--options", json.dumps(overrides), "--epoch-runner", "scripts.lr_tail_decay:run_epoch",
                        "--note", f"SM43 full: {len(users)} users x ({const} constant-LR + {decay} linear-decay epochs), "
                                  "equal reviews seen (data diversity vs epochs)"],
                       check=True, cwd=WT)
    run_p = EXP / name / "run/run.json"
    run = json.loads(run_p.read_text())
    run.update(train_users=users, training_reviews_per_epoch=reviews,
               split=run["split"].split("; SM43")[0] + f"; SM43 full subset of {len(users)} gate users (sm43_full_prepare.py)",
               lr=f"R01 schedule (AdamW peak 7e-4, 5-update warmup, constant) for epochs 1-{const}, then linear decay to 0 "
                  f"over epochs {const + 1}-{epochs} (scripts/lr_tail_decay.py)")
    run_p.write_text(json.dumps(run, indent=1))
    ex_p = EXP / name / "experiment.json"
    ex = json.loads(ex_p.read_text())
    ex.update(train_users=len(users), lr_tail_decay={"epochs": decay},
              sm43={"users": len(users), "epochs": epochs, "constant_epochs": const, "decay_epochs": decay,
                    "reviews_per_epoch": reviews, "reviews_total": reviews * epochs})
    ex_p.write_text(json.dumps(ex, indent=1))
per_decile_b = [sum(u in set(users_b) for u in d) for d in a_by_dec]
summary = {"a": {"experiment": "sm43full-a-1250u-11ep", "users": len(users_a), "epochs": CONST_A + DECAY_A, "constant_epochs": CONST_A,
                 "decay_epochs": DECAY_A, "reviews_per_epoch": ra, "reviews_total": ra * (CONST_A + DECAY_A), "pick_seed": 4343},
           "b": {"experiment": "sm43full-b-125u-110ep", "users": len(users_b), "epochs": CONST_B + DECAY_B, "constant_epochs": CONST_B,
                 "decay_epochs": DECAY_B, "reviews_per_epoch": rb, "reviews_total": rb * (CONST_B + DECAY_B), "pick_seed": seed_b,
                 "users_per_decile": per_decile_b, "deciles_with_13": thirteen},
           "ratio_total_b_over_a": rb * 10 / ra, "users_a": users_a, "users_b": users_b,
           "options_from": "scale-w64kd-gate-cached-ctx2048", "decay": "scripts/lr_tail_decay.py (linear 7e-4 -> 0 over the last N epochs)"}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sm43-full-users.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("users_")}))
