"""List git-ignored, untracked files under artifacts/ of the main checkout that also exist as tracked files on another local branch.

Jobs queued from the main checkout write their results here, while the committed copy lives on a track branch (worktree).
Each stray is classified against the first branch that tracks the same relative path:
  identical  same bytes as the committed blob
  subset     every line of the main copy is a line of the committed blob (text files only)
  different  anything else (kept; merge by hand)
Report-only by default. `--delete-identical-or-subset` removes only identical/subset copies. Nothing else is touched.
CPU only, read only apart from that flag. Run after a runbook finishes (docs/R01_PARALLEL_SESSIONS.md).
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".log", ".md"}
MAX_BYTES = 512 * 1024 * 1024


def git(*args, binary=False):
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.decode('utf-8', 'replace').strip()}")
    return done.stdout if binary else done.stdout.decode("utf-8", "replace")


def local_branches():
    current = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    return [name for name in git("for-each-ref", "--format=%(refname:short)", "refs/heads").split() if name != current]


def tracked_blobs(branch):
    """{relative path: blob id} of artifacts/ on a branch."""
    blobs = {}
    for line in git("ls-tree", "-r", "-z", branch, "--", "artifacts").split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            blobs[path] = meta.split()[2]
    return blobs


def blob_hash(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def classify(path, branch, blob):
    data = (ROOT / path).read_bytes()
    if blob_hash(data) == blob:
        return "identical"
    if Path(path).suffix.lower() in TEXT:
        committed = git("cat-file", "blob", blob, binary=True)
        theirs = set(committed.splitlines())
        mine = [line for line in data.splitlines() if line.strip()]
        if all(line in theirs for line in mine):
            return "subset"
        return f"different ({sum(line not in theirs for line in mine)} of {len(mine)} lines missing on {branch})"
    return "different (binary)"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--delete-identical-or-subset", action="store_true", help="remove main copies verified identical to or contained in the committed file")
    args = parser.parse_args()
    ignored = [p for p in git("ls-files", "-o", "-i", "--exclude-standard", "-z", "--", "artifacts").split("\0") if p]
    wanted = set(ignored)
    found = {}
    for branch in local_branches():
        for path, blob in tracked_blobs(branch).items():
            if path in wanted and path not in found:
                found[path] = (branch, blob)
    print(f"{len(ignored)} ignored untracked files under artifacts/; {len(found)} also tracked on another local branch")
    removable = 0
    for path in sorted(found):
        branch, blob = found[path]
        size = (ROOT / path).stat().st_size
        verdict = "different (too large to compare)" if size > MAX_BYTES else classify(path, branch, blob)
        redundant = verdict in ("identical", "subset")
        removable += redundant
        action = ""
        if redundant and args.delete_identical_or_subset:
            (ROOT / path).unlink()
            action = "  -> deleted"
        print(f"{verdict:12.12} {path}  [{branch}, {size} bytes]{action}" if redundant else f"{verdict}  {path}  [{branch}, {size} bytes]")
    print(f"{removable} redundant" + (" (deleted)" if args.delete_identical_or_subset else " (report only; pass --delete-identical-or-subset to remove)"))


if __name__ == "__main__":
    sys.exit(main())
