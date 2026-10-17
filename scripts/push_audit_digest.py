#!/usr/bin/env python
"""Digest of what was pushed since the last LLM audit, for the Haiku sensitive-content review (after the pre-push rules).

Reads the pushed ranges that scripts/prepush_audit.py logged (artifacts/push-audit/log.jsonl) after the marker
(artifacts/push-audit/llm-marker.json), or every commit on any branch since a date (--since, for a first baseline). Writes
artifacts/push-audit/digest-<time>.md: per commit the subject, the files, and the added lines of each text file (capped);
bulky data files (benchmark score rows, generated page data, > 200 KB) are listed by name and size only.

  .venv/Scripts/python.exe -I scripts/push_audit_digest.py [--since "2026-10-17 00:00"]
  .venv/Scripts/python.exe -I scripts/push_audit_digest.py --mark DIGEST   # after the review: advance the marker
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(r"C:\Users\jonat\workspace\anki-rwkv")
DIR = REPO / "artifacts" / "push-audit"
LOG, MARK = DIR / "log.jsonl", DIR / "llm-marker.json"
BULKY = re.compile(r"(heldout[^/]*\.jsonl|pages/data/factory\.json|\.npz|\.pth|\.pt|\.bin|\.png|\.jpg|\.gz|\.zip)$")
MAX_FILE_BYTES, MAX_LINES_PER_FILE, MAX_DIGEST = 200_000, 200, 250_000


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def ranges_from_log(after):
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines() if LOG.exists() else []:
        e = json.loads(line)
        if e["time"] <= after or e.get("blocked"):
            continue
        for r in e["ranges"]:
            if r["commits"]:
                base = r["from"] if r["from"] != "0" * 40 else None
                out.append((e["repo"], base, r["to"], e["time"]))
    return out


def commits(repo, base, tip):
    if base:
        return git(repo, "rev-list", "--reverse", f"{base}..{tip}").split()
    return git(repo, "rev-list", "--reverse", tip, "--not", "--remotes", f"--max-count=200").split()


def describe(repo, sha):
    head = git(repo, "show", "-s", "--format=%h %ad %an <%ae>%n%s", "--date=format:%m-%d %H:%M", sha).strip()
    parts = [f"### {head}"]
    for path in git(repo, "diff-tree", "--no-commit-id", "-r", "--root", "--diff-filter=AMR", "--name-only", sha).split("\n"):
        path = path.strip()
        if not path:
            continue
        size = git(repo, "cat-file", "-s", f"{sha}:{path}").strip()
        size = int(size) if size.isdigit() else 0
        if BULKY.search(path) or size > MAX_FILE_BYTES:
            parts.append(f"- `{path}` ({size / 1024:.0f} KB, content not shown)")
            continue
        added = [l[1:] for l in git(repo, "show", "--format=", "--unified=0", "--no-color", sha, "--", path).splitlines()
                 if l.startswith("+") and not l.startswith("+++")]
        more = f"\n  ... {len(added) - MAX_LINES_PER_FILE} more added lines" if len(added) > MAX_LINES_PER_FILE else ""
        parts.append(f"- `{path}` (+{len(added)} lines)\n```\n" + "\n".join(added[:MAX_LINES_PER_FILE]) + f"\n```{more}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help="baseline: every commit on any branch since this date")
    parser.add_argument("--mark", type=Path, help="advance the marker to the end of this reviewed digest")
    args = parser.parse_args()
    if args.mark:
        until = json.loads(args.mark.with_suffix(".json").read_text())["until"]
        MARK.write_text(json.dumps({"after": until}))
        print(f"marker -> {until}")
        return 0
    after = json.loads(MARK.read_text())["after"] if MARK.exists() else ""
    now = datetime.now().isoformat(timespec="seconds")
    if args.since:
        shas = git(REPO, "log", "--all", "--reverse", "--format=%H", f"--since={args.since}").split()
        work = [(str(REPO), shas)]
        until = now
    else:
        work, seen = [], set()
        rs = ranges_from_log(after)
        for repo, base, tip, _ in rs:
            new = [s for s in commits(repo, base, tip) if s not in seen]
            seen.update(new)
            work.append((repo, new))
        until = max([t for *_, t in rs], default=after or now)
    blocks = [describe(repo, sha) for repo, shas in work for sha in shas]
    if not blocks:
        print("nothing pushed since the last audit")
        return 0
    DIR.mkdir(parents=True, exist_ok=True)
    parts, cur = [], []
    for b in blocks:  # split into parts of at most MAX_DIGEST characters (one commit never split)
        if cur and sum(map(len, cur)) + len(b) > MAX_DIGEST:
            parts.append(cur); cur = []
        cur.append(b[:MAX_DIGEST])
    parts.append(cur)
    stamp = now.replace(":", "")
    for i, part in enumerate(parts, 1):
        out = DIR / f"digest-{stamp}-{i}of{len(parts)}.md"
        head = f"# Pushed commits to audit, part {i}/{len(parts)} ({len(part)} commits, after {after or args.since})"
        out.write_text(head + "\n\n" + "\n\n".join(part), encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps({"until": until, "commits": len(part)}))
        print(f"{out} ({len(part)} commits, {sum(map(len, part)):,} characters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
