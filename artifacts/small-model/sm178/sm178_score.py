"""Today's models on donor data (SM178): score the release-candidate family on the donated users.

The held-out scorer (`scripts/small_model_heldout.py` of the sm164 worktree, commit c30a0070, unchanged) reads the 8k
run's folder (run.json, label-db) and the 10k dataset. This wrapper points it at the donations dataset the way the
converter's smoke test (SM177) did: it replaces `compress_common.run_info / data_path / RUN` and the pool, then calls the
scorer's own main() with the given arguments. Same protocol as the 732 set: answer-time layout, `--inputs e2s-query`,
`--deleted-cards neighbour`, FSRS-6 benchmark test reviews (`rwkv/find_equalize_test_reviews.process`).

  labels: python sm178_score.py labels LABEL_RUN_DIR PROCESSES USERS_FILE      (CPU; writes LABEL_RUN_DIR/label-db)
  score : python sm178_score.py score  LABEL_RUN_DIR -- <scorer arguments>     (--users-file required)
  users : python sm178_score.py users                                         (writes the users files next to this file)
"""
import json
import os
import sys
from pathlib import Path

MAIN = Path(r"C:\Users\jonat\workspace\anki-rwkv")
SCORER_WT = Path(r"C:\Users\jonat\workspace\wt\sm164")
DATA = MAIN / "data" / "donations-20261126-timefix-answer-time"   # 11-27: pre-2006 reviews dropped (sm178_timefix.py)
SOURCES = MAIN / "data" / "donations-20261126-timefix" / "sources.parquet"
HERE = Path(__file__).resolve().parent
SM177_SMOKE = [1000197, 1000132, 1000104, 1000046, 1000164, 1000122, 1000084, 1000117, 1000170, 1000210, 1000121,
               1000161, 1000019, 1000125, 1000136, 1000195, 1000020, 1000094, 1000213, 1000118]


def counts():
    import pandas as pd
    users = sorted(int(p.name.split("=")[1]) for p in (DATA / "revlogs").iterdir())
    return {str(u): len(pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["card_id"])) for u in users}


def label_job(args):
    toml, user = args
    sys.argv = [sys.argv[0], "--config", toml]
    from rwkv import find_equalize_test_reviews as labels
    labels.process(user)
    return user


def label_init(toml):
    os.chdir(MAIN)
    sys.path.insert(0, str(MAIN))
    sys.argv = [sys.argv[0], "--config", toml]
    from rwkv import find_equalize_test_reviews as labels
    labels.set_low_priority()


def main():
    stage = sys.argv[1]
    if stage == "users":
        c = counts()
        users = sorted(int(u) for u in c)
        (HERE / "users-all.json").write_text(json.dumps({"users": users}) + "\n")
        (HERE / "users-sm177-smoke.json").write_text(json.dumps({"users": SM177_SMOKE}) + "\n")
        mid = sorted(users, key=lambda u: c[str(u)])
        smoke3 = [u for u in SM177_SMOKE if 20_000 <= c[str(u)] <= 120_000][:3]
        (HERE / "users-smoke3.json").write_text(json.dumps({"users": smoke3}) + "\n")
        print(len(users), sum(c.values()), smoke3, [c[str(u)] for u in smoke3], mid[-3:])
        return
    run_dir = Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "labels":
        import multiprocessing
        processes, users_file = int(sys.argv[3]), Path(sys.argv[4])
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        toml = run_dir / "labels.toml"
        toml.write_text(f'DATA_PATH = "{DATA.as_posix()}"\nLABEL_FILTER_LMDB_PATH = "{(run_dir / "label-db").as_posix()}"\n'
                        'LABEL_FILTER_LMDB_SIZE = 4_000_000_000\nALGO = "FSRS-6"\nSHORT = true\nSECS = true\n'
                        f'USER_START = 1\nUSER_END = 1\nPROCESSES = {processes}\n')
        c = counts()
        users = json.loads(users_file.read_text())["users"]
        big = sorted(users, key=lambda u: -c[str(u)])   # largest first, so the pool ends on small users
        with multiprocessing.Pool(processes, initializer=label_init, initargs=(str(toml),), maxtasksperchild=8) as pool:
            for done, _ in enumerate(pool.imap_unordered(label_job, [(str(toml), u) for u in big]), 1):
                if done % 10 == 0 or done == len(big):
                    print(f"[sm178 labels] {done}/{len(big)}", flush=True)
        print("[sm178 labels] DONE", flush=True)
        return
    assert stage == "score" and sys.argv[3] == "--"
    c = counts()
    sys.path.insert(0, str(SCORER_WT))
    from scripts import small_model_heldout as h  # noqa: E402

    def donor_users(cnt, pool=464):   # the scorer's decile interleave, on the donor users
        users = sorted((int(u) for u in cnt), key=lambda u: (cnt[str(u)], u))
        deciles = [users[len(users) * d // 10: len(users) * (d + 1) // 10] for d in range(10)]
        return [g[r] for r in range(max(len(g) for g in deciles)) for g in deciles if r < len(g)]

    h.cc.run_info = lambda: {"review_counts": c, "dataset_path": str(DATA)}
    h.cc.data_path = lambda run=None: DATA
    h.cc.RUN = run_dir
    h.heldout_users = donor_users
    if "--users-file" not in sys.argv:
        raise SystemExit("--users-file is required")
    sys.argv = [sys.argv[0]] + sys.argv[4:]
    h.main()


if __name__ == "__main__":
    main()
