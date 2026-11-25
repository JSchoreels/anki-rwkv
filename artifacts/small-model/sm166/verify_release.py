#!/usr/bin/env python
"""Weekend family release (SM166): verify the built assets from a clean extract (NumPy only, CPU).

    python -X utf8 artifacts/small-model/sm166/verify_release.py --dist work/release-2026-11-28/dist --scratch <empty dir>

1. SHA256SUMS matches every asset;
2. each zip is extracted into its own new, empty folder; it must hold exactly one top-level folder named after the zip,
   WEIGHTS-NOTICE.md and INTEGRATION.md, test vectors of the synthetic users 900001 / 900002 only, and no logs;
3. in the extracted folder: testvectors/check_testvectors.py and testvectors/check_time_vectors.py must print
   "RESULT: PASS" (each run as its own process, cwd = the extract root);
4. writes logs/verify-<zip>.log and logs/verify-summary.json next to this script.
"""
from argparse import ArgumentParser
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    dist, scratch = Path(args.dist), Path(args.scratch)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    summary = {"checked": time.strftime("%Y-%m-%d %H:%M"), "python": sys.version.split()[0], "sums": {}, "zips": {}}
    for line in (dist / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        summary["sums"][name] = sha256(dist / name) == digest
    if not all(summary["sums"].values()):
        raise SystemExit(f"SHA256SUMS mismatch: {summary['sums']}")
    for z in sorted(dist.glob("*.zip")):
        if args.only and z.stem not in args.only:
            continue
        target = scratch / z.stem
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        with zipfile.ZipFile(z) as f:
            names = f.namelist()
            tops = {n.split("/", 1)[0] for n in names}
            if tops != {z.stem}:
                raise SystemExit(f"{z.name}: top-level entries {tops}")
            f.extractall(target)
        pkg = target / z.stem
        files = sorted(str(p.relative_to(pkg)).replace("\\", "/") for p in pkg.rglob("*") if p.is_file())
        npz = [n for n in files if n.endswith(".npz")]
        allowed = {f"testvectors/{p}user_{u}.npz" for p in ("", "time_") for u in (900001, 900002)}
        problems = [n for n in npz if n not in allowed] + [n for n in files if n.endswith((".log", ".safetensors"))]
        for need in ("WEIGHTS-NOTICE.md", "INTEGRATION.md", "time_head.f32.bin", "time_head.int8.bin",
                     "time_head.json", "weights.f32.bin", "weights.int8.bin"):
            if need not in files:
                problems.append(f"missing {need}")
        rec = {"files": len(files), "problems": problems, "checks": {}}
        log = [f"== {z.name} ({z.stat().st_size:,} B), extracted to {pkg}", f"files: {files}"]
        for check in ("check_testvectors.py", "check_time_vectors.py"):
            started = time.perf_counter()
            r = subprocess.run([sys.executable, "-I", "-X", "utf8", str(pkg / "testvectors" / check)], cwd=str(target),
                               capture_output=True, text=True)
            out = (r.stdout + r.stderr).strip()
            last = [l for l in out.splitlines() if l.startswith("RESULT:")]
            rec["checks"][check] = {"exit": r.returncode, "result": last[-1] if last else "no RESULT line",
                                    "seconds": round(time.perf_counter() - started, 1)}
            log += [f"-- python -I testvectors/{check} (exit {r.returncode}, "
                    f"{rec['checks'][check]['seconds']} s)", out]
        rec["pass"] = not problems and all(c["exit"] == 0 and c["result"].startswith("RESULT: PASS")
                                           for c in rec["checks"].values())
        (logs / f"verify-{z.stem}.log").write_text("\n".join(log) + "\n", encoding="utf-8", newline="\n")
        summary["zips"][z.name] = rec
        print(f"[sm166-verify] {z.name}: {'PASS' if rec['pass'] else 'FAIL'} "
              f"{ {k: v['result'] for k, v in rec['checks'].items()} } problems {problems}", flush=True)
    summary["all_pass"] = all(r["pass"] for r in summary["zips"].values())
    (logs / "verify-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("ALL PASS" if summary["all_pass"] else "FAILURES", flush=True)
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
