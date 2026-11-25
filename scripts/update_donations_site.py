"""Refresh the data campaign page from the donation folder and push it (meant to run every 4 hours).

Steps: fast-forward main, rerun the inventory only if the donation folder changed (publish_donations_page.py),
then commit ONLY pages/data/donations.json if it changed, merge origin/main and push. Prints one summary line.

Exit codes: 0 = nothing to do or pushed; 1 = failure (message printed; nothing half-committed is pushed).
The donation folder is read-only here; extracted collections go to a temp folder that the publisher deletes.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
DATA = "pages/data/donations.json"
HIST = "pages/data/donations-history.json"
IDENT = ["-c", "user.name=Jonathan Schoreels", "-c", "user.email=jonathan.schoreels@gmail.com"]


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def main():
    try:
        if git("branch", "--show-current") != "main":
            print("FAIL: the anki-rwkv checkout is not on main")
            return 1
        git("fetch", "-q")
        git("merge", "-q", "--ff-only", "origin/main", check=False)
        pub = subprocess.run([str(PY), "-X", "utf8", str(REPO / "scripts" / "publish_donations_page.py")], cwd=REPO,
                             capture_output=True, text=True, encoding="utf-8")
        if pub.returncode:
            print("FAIL: publish_donations_page.py: " + (pub.stderr or pub.stdout).strip()[-300:])
            return 1
        check = subprocess.run([str(PY), "-X", "utf8", str(REPO / "scripts" / "publish_donations_page.py"), "--check"],
                               cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        if check.returncode:
            print("FAIL: privacy/consistency check: " + check.stdout.strip()[-300:])
            return 1
        if not git("status", "--porcelain", "--", DATA, HIST):
            print("UNCHANGED: " + pub.stdout.strip().splitlines()[-1])
            return 0
        git(*IDENT, "commit", "-q", "-m",
            "Pages: data campaign stats refreshed\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
            "--", DATA, HIST)
        git("fetch", "-q")
        git("merge", "-q", "origin/main")
        git("push", "-q", "origin", "main")
        print(f"PUSHED {git('rev-parse', '--short', 'HEAD')}: " + pub.stdout.strip().splitlines()[-1])
        return 0
    except Exception as exc:  # noqa: BLE001 - report one line and stop
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
