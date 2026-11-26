"""Make sure the published site matches main: if the newest "Deploy R01 pages" run failed, start a fresh deploy.

GitHub Pages deploys sometimes fail with a transient artifact error ("Fetching artifact metadata failed"). Re-running
that old run is wrong when newer pushes landed since (it would republish a stale snapshot), so this dispatches a new
run of the workflow on current main instead (workflow_dispatch), which always deploys the newest files.

Prints one line: OK / RUNNING / DISPATCHED / FAIL. Needs the gh CLI (already logged in). Safe to call often: it only
dispatches when the newest run completed with a failure and no dispatched run is already queued or running.
"""

import json
import subprocess
import sys

REPO = "JSchoreels/anki-rwkv"
WORKFLOW = "pages.yml"


def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:300])
    return r.stdout


def main():
    try:
        runs = json.loads(gh("run", "list", "--repo", REPO, "--workflow", WORKFLOW, "--limit", "5",
                             "--json", "databaseId,status,conclusion,event,createdAt"))
        if not runs:
            print("OK: no runs")
            return 0
        newest = runs[0]
        if newest["status"] != "completed":
            print(f"RUNNING: {newest['databaseId']} ({newest['event']})")
            return 0
        if newest["conclusion"] == "success":
            print(f"OK: {newest['databaseId']}")
            return 0
        gh("workflow", "run", WORKFLOW, "--repo", REPO, "--ref", "main")
        print(f"DISPATCHED: newest run {newest['databaseId']} ended {newest['conclusion']}; fresh deploy of main started")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
