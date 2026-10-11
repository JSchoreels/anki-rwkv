"""Record Git identity and preserve the source used by an experiment (no GPU)."""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ("rwkv", "expertium", "scripts", "configs", "pyproject.toml", "uv.lock",
                "justfile", "upstream.json")


def capture_source(root=ROOT):
    """Capture once before training; include untracked source, exclude datasets.

    A dirty HEAD identifies the base commit only. The archive and its content
    manifest identify the actual source, including staged and unstaged edits.
    Git/snapshot errors propagate rather than silently claiming reproducibility.
    """
    root = Path(root).resolve()

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    commit = git("rev-parse", "--verify", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard",
                "--", *SOURCE_PATHS).decode("utf-8").split("\0")
    files = {}
    for name in sorted(set(paths) - {""}):
        path = root / name
        if not path.exists():  # Tracked deletion; absent from the source snapshot.
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source path resolves outside the repository: {name}")
        files[name] = path.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    identity = {"git_commit": commit, "source_paths": list(SOURCE_PATHS), "files": manifest}
    encoded = json.dumps(identity, sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    folder = root / "artifacts/experiment-sources" / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "source.zip"
    if not archive.exists():
        temporary = folder / f"source-{uuid4().hex}.tmp"
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as snapshot:
                for name, data in files.items():
                    snapshot.writestr(name, data)
                snapshot.writestr("source-manifest.json", encoded)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    return {"captured_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit, "git_dirty": bool(status),
            "git_status_porcelain_z": status.decode("utf-8"),
            "source_tree_sha256": fingerprint,
            "source_archive": archive.relative_to(root).as_posix(),
            "source_paths": list(SOURCE_PATHS)}


@lru_cache(maxsize=1)
def process_provenance():
    """Reuse the initial process snapshot; restarting captures new source edits."""
    return capture_source()


def provenance_rows(report, suite, link_prefix="../"):
    """Render only recorded per-run revisions; never infer legacy ones from HEAD."""
    grouped = {}
    for item in report.get("results", []):
        provenance = item.get("provenance") or item.get("result", {}).get("provenance", {})
        key = (provenance.get("git_commit"), provenance.get("git_dirty"),
               provenance.get("source_tree_sha256"), provenance.get("source_archive"))
        grouped.setdefault(key, []).append(item)
    rows = []
    for (commit, dirty, fingerprint, archive), items in grouped.items():
        names = ", ".join(sorted({f"{i.get('model', '—')}/{i.get('variant', i.get('name', '—'))}"
                                   for i in items}))
        revision = f"`{commit}`" if commit else "— (not recorded)"
        state = "dirty" if dirty is True else "clean" if dirty is False else "—"
        source = (f"[`{fingerprint[:12]}`]({link_prefix}{archive})"
                  if fingerprint and archive else "—")
        rows.append(f"| {suite} | {len(items)} | {names} | {revision} | {state} | {source} |")
    return rows
