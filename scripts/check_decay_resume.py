#!/usr/bin/env python
"""CPU check of the resumable decay (scripts/expertium_decay.py run after an interruption), no GPU:
  1. a fake decay stopped at a snapshot: resume_point keeps the prepared protocol, sets the attempt aside, trains only
     the users not done, from the snapshot's weights and optimizer state;
  2. the learning rate and the KD "decaying" switch of the resumed run continue the uninterrupted schedule exactly;
  3. a second interruption adds its done users to the first; a stop before any snapshot keeps the previous resume point;
  4. an attempt without any snapshot restarts from the prepared protocol; a finished decay is not retrained; a run that
     finished training but stopped in evaluation is refused.
  python scripts/check_decay_resume.py   (prints PASS lines, exit 1 on a failure)"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import expertium_decay as D  # noqa: E402
from scripts.expertium_options import CatchUp  # noqa: E402

TRACK = "check-track"
FAIL = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(detail) if detail and not ok else ''}", flush=True)
    if not ok:
        FAIL.append(name)


def fake_decay(out, users):
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rows = [{"user_id": u, "available_reviews": n, "train_range": [0, n], "validation_range": [0, n]} for u, n in users]
    full_run = {"epoch": 10, "evaluate": False, "init_master": "source/master.pth", "update_offset": 1000,
                "lr_schedule": {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0},
                "optimizer_state": "fresh", "snapshot_users": 2, "control_dir": (out / "control").relative_to(ROOT).as_posix()}
    D.atomic_json(out / "decay.json", {"track": TRACK, "options": {}})
    D.atomic_json(out / "protocol.json", {"rows": rows, "evaluation_rows": [], "full_run": full_run})
    return rows


def interrupt(out, done, updates_end, status="stopped", snapshot=True, finished_training=False):
    case = out / D.MODEL / TRACK
    case.mkdir(parents=True)
    result = {"status": status}
    if finished_training:
        result["update_offset_end"] = updates_end
        (case / "optimizer.pth").write_bytes(b"x")
    D.atomic_json(case / "result.json", result)
    if snapshot:
        folder = case / "snapshots" / f"users-{len(done):05d}"
        folder.mkdir(parents=True)
        (folder / "master.pth").write_bytes(b"m")
        (folder / "optimizer.pth").write_bytes(b"o")
        meta = {"reason": "periodic", "done_users": done, "update_offset_end": updates_end,
                "folder": folder.relative_to(ROOT).as_posix()}
        D.atomic_json(folder / "snapshot.json", meta)
        D.atomic_json(case / "snapshots" / "latest.json", meta)


def main():
    out = ROOT / "artifacts/small-model/smoke/check-decay-resume"
    users = [(11, 300), (12, 500), (13, 200), (14, 400), (15, 600)]
    rows = fake_decay(out, users)
    total = sum(n for _, n in users)
    prepared = json.loads((out / "protocol.json").read_text())
    whole = CatchUp({}, prepared["full_run"], rows)

    interrupt(out, [11, 12], 1040)
    D.resume_point(out, {"track": TRACK})
    p = json.loads((out / "protocol.json").read_text())
    fr = p["full_run"]
    record("prepared protocol kept", json.loads((out / "protocol.prepared.json").read_text()) == prepared)
    record("attempt set aside", (out / "attempts/attempt-1" / D.MODEL / TRACK / "result.json").exists()
           and not (out / D.MODEL).exists())
    record("only users not done", [r["user_id"] for r in p["rows"]] == [13, 14, 15], p["rows"])
    record("from the snapshot", fr["init_master"].endswith("users-00002/master.pth")
           and fr["init_optimizer"].endswith("users-00002/optimizer.pth") and fr["optimizer_state"] == "snapshot", fr)
    record("update offset continues", fr["update_offset"] == 1040, fr["update_offset"])
    record("lr_resume", fr["lr_resume"] == {"total_reviews": total, "reviews_offset": 800}, fr["lr_resume"])

    resumed = CatchUp({}, fr, p["rows"])
    same = all(abs(resumed.learning_rate(k, 1040, r) - whole.learning_rate(k + 40, 1000, r + 800)) < 1e-15
               for k, r in ((0, 0), (5, 150), (20, 900), (39, 1199)))
    record("lr continues the uninterrupted schedule", same)
    record("lr reaches 0 at the end", resumed.learning_rate(60, 1040, total - 800) == 0.0)
    record("lr at resume start = lr at the snapshot", resumed.learning_rate(0, 1040, 0) == whole.learning_rate(40, 1000, 800)
           and 0 < resumed.learning_rate(0, 1040, 0) < 7e-4)
    record("KD decaying switch uses the whole decay", resumed.schedule.decaying(0 + resumed.reviews_offset))
    try:
        CatchUp({}, fr, p["rows"][:-1])
        record("lr_resume refuses a row set that does not add up", False)
    except ValueError:
        record("lr_resume refuses a row set that does not add up", True)

    interrupt(out, [13], 1060)
    D.resume_point(out, {"track": TRACK})
    p = json.loads((out / "protocol.json").read_text())
    record("second interruption adds its done users", [r["user_id"] for r in p["rows"]] == [14, 15]
           and p["full_run"]["lr_resume"]["reviews_offset"] == 1000 and p["full_run"]["update_offset"] == 1060, p["full_run"])
    record("second attempt kept", (out / "attempts/attempt-2").exists())

    interrupt(out, [], 0, snapshot=False)
    D.resume_point(out, {"track": TRACK})
    p2 = json.loads((out / "protocol.json").read_text())
    record("stop before any new snapshot keeps the previous resume point", p2["rows"] == p["rows"]
           and p2["full_run"]["init_master"] == p["full_run"]["init_master"], p2["full_run"])

    interrupt(out, [], 1100, status="complete", snapshot=False)
    record("finished decay is not retrained", D.resume_point(out, {"track": TRACK}) == "complete")
    shutil.rmtree(out / D.MODEL)

    interrupt(out, [], 1100, status="stopped", snapshot=False, finished_training=True)
    try:
        D.resume_point(out, {"track": TRACK})
        record("stopped in evaluation is refused", False)
    except RuntimeError:
        record("stopped in evaluation is refused", True)

    import torch
    from scripts.muon_optimizer import expertium_iter53

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.randn(4, 4))
            self.bias = torch.nn.Parameter(torch.zeros(4))
    optimizer = expertium_iter53(Toy())
    optimizer.load_state_dict(optimizer.state_dict())
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-4
    inner = optimizer.adamw.param_groups + optimizer.muon.param_groups
    record("Muon optimizer: lr writes reach the loaded groups", len(optimizer.muon.param_groups) > 0
           and all(group["lr"] == 1.25e-4 for group in inner))

    rows = fake_decay(out, users)
    interrupt(out, [], 0, status="failed", snapshot=False)
    D.resume_point(out, {"track": TRACK})
    record("no snapshot at all: restart from the prepared protocol",
           json.loads((out / "protocol.json").read_text()) == prepared and (out / "attempts/attempt-1").exists())
    shutil.rmtree(out, ignore_errors=True)
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
