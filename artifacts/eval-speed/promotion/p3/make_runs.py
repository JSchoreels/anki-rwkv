"""EV07 smoke: two tiny full runs (fast_evaluation false/true) from the 8k run's splits; small label DBs copied key by key."""
import json, random, sys
from pathlib import Path
import lmdb

MAIN = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\full-run\run-8k-20261013")
HERE = Path(__file__).resolve().parent
config = json.loads((MAIN / "run.json").read_text())
rng = random.Random(20261014)
train = sorted(rng.sample(config["train_users"], 50))
validation = sorted(rng.sample(config["validation_users"], 80))
final_test = sorted(rng.sample(config["final_test_users"], 20))
small = dict(config)
small.update(created_utc="2026-10-14T00:00:00+00:00", epochs=1, dataset_path=str(Path(r"C:\Users\jonat\workspace\anki-rwkv") / config["dataset_path"]),
             train_users=train, validation_users=validation, final_test_users=final_test,
             quick_validation_users=validation[:20], preserved_final_test_users=[u for u in config["preserved_final_test_users"] if u in final_test],
             training_reviews_per_epoch=sum(config["review_counts"][str(u)] for u in train),
             smoke_note="EV07 smoke derived from run-8k-20261013 (50 train, 80 validation, 20 final-test users, 1 epoch)")
source = lmdb.open(str(MAIN / "label-db"), 10_000_000_000, readonly=True, lock=False)
for name, fast in (("run-reference", False), ("run-fast", True)):
    run = HERE / name
    run.mkdir(exist_ok=False)
    value = dict(small)
    if fast:
        value["fast_evaluation"] = True
    (run / "run.json").write_text(json.dumps(value, indent=1) + "\n")
    target = lmdb.open(str(run / "label-db"), 1 << 30)
    copied = 0
    with source.begin() as read, target.begin(write=True) as write:
        for user in validation + final_test:
            for key in (f"{user}_review_ths", f"{user}_rmse_bins"):
                data = read.get(key.encode())
                if data is not None:
                    write.put(key.encode(), data)
                    copied += 1
    target.close()
    print(name, "label keys", copied)
print(json.dumps({"train_reviews": small["training_reviews_per_epoch"],
                  "eval_reviews": sum(config["review_counts"][str(u)] for u in validation + final_test)}))
