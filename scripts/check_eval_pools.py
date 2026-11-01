"""Checks for the evaluation pools: the default 464 order is unchanged, pools are fork-unseen, disjoint from the gate
training users and the sealed test users, nested, and the committed eval-users files match. Prints ALL OK."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1]
from scripts.eval_pools import gate_train_users, pool_users, split_info  # noqa: E402
from scripts.small_model_heldout import TIERS, _screen_eval_users, heldout_users  # noqa: E402


def main():
    _, split, tiers = split_info()
    counts = {str(k): v for k, v in split["review_counts"].items()}
    # default path identical: the original expression, the same ordering
    old = sorted(u for u in _screen_eval_users(json.loads(TIERS.read_text())) if u < 5000)
    old.sort(key=lambda u: (counts[str(u)], u))
    dec = [old[len(old) * d // 10: len(old) * (d + 1) // 10] for d in range(10)]
    expected = []
    for rank in range(max(len(g) for g in dec)):
        expected += [g[rank] for g in dec if rank < len(g)]
    assert heldout_users(counts) == heldout_users(counts, 464) == expected and len(expected) == 464, "464 order changed"
    gate_train, _ = gate_train_users(split)
    sealed = set(split["final_test_users"])
    pools = {p: pool_users(p) for p in (464, 732, 2225)}
    for p, users in pools.items():
        assert len(users) == p == len(set(users)), f"pool {p} size {len(users)}"
        assert all(u < 5000 for u in users), f"pool {p} has fork-seen users"
        assert not set(users) & gate_train, f"pool {p} overlaps the gate training users"
        assert not set(users) & sealed, f"pool {p} overlaps final_test_users"
        assert sorted(heldout_users(counts, p)) == users, f"pool {p} ordering changed the set"
    assert set(pools[464]) < set(pools[732]) < set(pools[2225]), "pools not nested"
    assert set(pools[732]) == {u for u in split["validation_users"] if u < 5000}
    for p in (732, 2225):
        doc = json.loads((ROOT / "artifacts/small-model" / f"eval-users-{p}.json").read_text())
        assert doc["users"] == pools[p], f"eval-users-{p}.json is stale"
        print(f"pool {p}: {len(doc['users'])} users, {sum(counts[str(u)] for u in doc['users']):,} reviews")
    print(f"pool 464: {sum(counts[str(u)] for u in pools[464]):,} reviews")
    print("ALL OK")


if __name__ == "__main__":
    main()
