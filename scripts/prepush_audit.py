#!/usr/bin/env python
"""Pre-push audit: blocks a push that would publish sensitive files or content (installed as the shared pre-push hook).

Git calls the hook with `<remote name> <remote url>` and one stdin line per ref: `<local ref> <local sha> <remote ref>
<remote sha>`. For every commit that the remote does not have yet, the audit checks the added files and added lines:

- never published: docs/GPU_OPTIMIZATIONS_LOG.md, .claude/, Anki collections and packages (.anki2 / .anki21 / .anki21b /
  .apkg / .colpkg, collection.media), raw revlog data (.parquet, .sqlite, .db), env files and private keys;
- own collection (aggregate only): .npz files, or rows with card_id / review_th / per-review arrays, under a path that names
  a collection;
- secrets in added lines: private key blocks and common token formats (Anthropic, OpenAI, GitHub, AWS, Slack, Hugging Face,
  Google), plus the private patterns in artifacts/push-audit/private-patterns.txt (not committed: they would name what
  they hide);
- commit metadata: a warning (no block) when the author or committer e-mail is not the usual public identity;
- files over 90 MB (GitHub refuses 100 MB); a warning, not a block, above 40 MB.

Exit 1 blocks the push and prints what was found. Every run is logged to artifacts/push-audit/log.jsonl (not committed) with
the pushed ranges, so the after-push Haiku audit knows what left the machine.

  .venv/Scripts/python.exe -I scripts/prepush_audit.py --range A..B     # audit a range by hand (no push)
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

LOG = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\push-audit\log.jsonl")
PRIVATE = LOG.parent / "private-patterns.txt"
PUBLIC_EMAIL = "jonathan.schoreels@gmail.com"
BOT_EMAIL = re.compile(r"(noreply|no-reply|^codex@openai.(com|invalid)$|^git@stash$)", re.I)  # tools, not people
ZERO = "0" * 40
MAX_BYTES = 90 * 1024 * 1024  # GitHub refuses 100 MB
WARN_BYTES = 40 * 1024 * 1024
BLOCKED_PATHS = [
    (re.compile(r"(^|/)docs/GPU_OPTIMIZATIONS_LOG\.md$"), "local optimisation log (never committed)"),
    (re.compile(r"(^|/)\.claude/"), "Claude Code local settings"),
    (re.compile(r"\.(anki2|anki21|anki21b|apkg|colpkg)$", re.I), "Anki collection or package"),
    (re.compile(r"(^|/)collection\.media(/|$)", re.I), "Anki media folder"),
    (re.compile(r"\.(parquet|sqlite|sqlite3|db)$", re.I), "raw review data / database"),
    (re.compile(r"(^|/)\.env(\.|$)"), "environment file"),
    (re.compile(r"(^|/)(id_rsa|id_ed25519|id_ecdsa|id_dsa)$|\.(pem|pfx|p12|key)$", re.I), "private key"),
    (re.compile(r"(^|/)(credentials|secrets)\.(json|ya?ml)$", re.I), "credentials file"),
    (re.compile(r"(^|/)Anki2/User 1/", re.I), "file from the owner's Anki profile"),
]
COLLECTION_PATH = re.compile(r"collection", re.I)
COLLECTION_ROW = re.compile(r'"(card_id|cid|review_th|note_id|nid|p_ahead|p_instant|predictions)"\s*:')
SECRETS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "Anthropic API key"),
    (re.compile(r"\bsk-(proj-)?[A-Za-z0-9]{32,}"), "OpenAI-style API key"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "Google API key"),
]


def git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=check).stdout


def commits(local, remote):
    if remote != ZERO:
        return git("rev-list", f"{remote}..{local}").split()
    return git("rev-list", local, "--not", "--remotes").split()  # new branch: only what no remote has


def private_patterns():
    try:
        lines = PRIVATE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [(re.compile(l.strip(), re.I), "private pattern (artifacts/push-audit/private-patterns.txt)")
            for l in lines if l.strip() and not l.startswith("#")]


def audit(shas):
    found = []
    secrets = SECRETS + private_patterns()
    for sha in shas:
        for who, email in zip(("author", "committer"), git("show", "-s", "--format=%ae%n%ce", sha).split()):
            if email != PUBLIC_EMAIL and not BOT_EMAIL.search(email):
                print(f"[push-audit] warning: {sha[:10]} {who} e-mail {email} is not {PUBLIC_EMAIL} (not blocked)", file=sys.stderr)
        for line in git("diff-tree", "--no-commit-id", "-r", "--root", "--diff-filter=AMR", "-z", "--name-only", sha).split("\0"):
            path = line.strip()
            if not path:
                continue
            for rx, why in BLOCKED_PATHS:
                if rx.search(path):
                    found.append((sha, path, why))
            size = git("cat-file", "-s", f"{sha}:{path}", check=False).strip()
            if size.isdigit() and int(size) > MAX_BYTES:
                found.append((sha, path, f"{int(size) / 2**20:.0f} MB file (limit 90 MB)"))
            elif size.isdigit() and int(size) > WARN_BYTES:
                print(f"[push-audit] warning: {path} is {int(size) / 2**20:.0f} MB (GitHub refuses 100 MB)", file=sys.stderr)
            if COLLECTION_PATH.search(path) and path.lower().endswith(".npz"):
                found.append((sha, path, "per-review arrays from a collection"))
        patch = git("show", "--format=", "--unified=0", "--no-color", "--no-ext-diff", sha)
        path = None
        for line in patch.splitlines():
            if line.startswith("+++ "):
                path = line[6:] if line.startswith("+++ b/") else None
                continue
            if not line.startswith("+") or path is None:
                continue
            for rx, why in secrets:
                if rx.search(line):
                    found.append((sha, path, why))
            if COLLECTION_PATH.search(path) and COLLECTION_ROW.search(line):
                found.append((sha, path, "per-card / per-review fields in an own-collection file (aggregate only)"))
    return sorted(set(found))


def log(entry):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("remote", nargs="?")
    parser.add_argument("url", nargs="?")
    parser.add_argument("--range", help="audit A..B by hand instead of reading the hook's stdin")
    args = parser.parse_args()
    if args.range:
        a, b = args.range.split("..")
        refs = [("manual", git("rev-parse", b).strip(), "manual", git("rev-parse", a).strip())]
    else:
        refs = [tuple(line.split()) for line in sys.stdin if len(line.split()) == 4]
    shas, ranges = [], []
    for local_ref, local, remote_ref, remote in refs:
        if local == ZERO:
            continue  # branch deletion: nothing published
        new = commits(local, remote)
        shas += new
        ranges.append({"ref": remote_ref, "from": remote, "to": local, "commits": len(new)})
    found = audit(list(dict.fromkeys(shas)))
    repo = git("rev-parse", "--show-toplevel").strip()
    log({"time": datetime.now().isoformat(timespec="seconds"), "repo": repo, "remote": args.remote, "ranges": ranges,
         "blocked": bool(found), "findings": [{"commit": s[:10], "path": p, "why": w} for s, p, w in found]})
    if found:
        print("[push-audit] push BLOCKED: sensitive content in the commits being pushed", file=sys.stderr)
        for sha, path, why in found:
            print(f"  {sha[:10]} {path}: {why}", file=sys.stderr)
        print("[push-audit] remove it from those commits (amend/rebase locally), then push again", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
