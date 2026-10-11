"""Exercise source traceability in an isolated temporary Git repository; no GPU."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import capture_source, provenance_rows


def main():
    temporary_root = ROOT / "artifacts/check-temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=temporary_root) as folder:
        repo = Path(folder)

        def git(*args):
            return subprocess.run(["git", "-C", str(repo), *args], check=True,
                                  capture_output=True).stdout.decode().strip()

        git("init", "-q")
        git("config", "user.name", "Provenance check")
        git("config", "user.email", "provenance-check@example.invalid")
        git("config", "core.autocrlf", "false")
        (repo / ".gitignore").write_text("artifacts/\ndata/\n", encoding="utf-8")
        (repo / "scripts").mkdir()
        (repo / "expertium").mkdir()
        (repo / "expertium/model.py").write_bytes(b"small architecture\n")
        source = repo / "scripts/example.py"
        source.write_text("original\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-qm", "Fixture")
        original_head = git("rev-parse", "HEAD")
        clean = capture_source(repo)
        assert clean["git_commit"] == original_head and clean["git_dirty"] is False
        assert len(clean["git_commit"]) == 40
        source.write_text("staged\n", encoding="utf-8")
        git("add", "scripts/example.py")
        source.write_bytes(b"actual unstaged source\n")
        (repo / "scripts/untracked.py").write_bytes(b"new local code\n")
        (repo / "data").mkdir()
        (repo / "data/private.txt").write_text("must not be archived", encoding="utf-8")
        dirty = capture_source(repo)
        assert dirty["git_commit"] == original_head and dirty["git_dirty"] is True
        assert dirty["source_tree_sha256"] != clean["source_tree_sha256"]
        archive_path = repo / dirty["source_archive"]
        with ZipFile(archive_path) as archive:
            assert archive.read("scripts/example.py") == b"actual unstaged source\n"
            assert archive.read("scripts/untracked.py") == b"new local code\n"
            assert archive.read("expertium/model.py") == b"small architecture\n"
            assert not any(name.startswith(("artifacts/", "data/")) for name in archive.namelist())
            encoded = archive.read("source-manifest.json")
            manifest = json.loads(encoded)
            for name, digest in manifest["files"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert hashlib.sha256(encoded).hexdigest() == dirty["source_tree_sha256"]
        unchanged = capture_source(repo)
        assert unchanged["source_archive"] == dirty["source_archive"]
        assert len(list(archive_path.parent.glob("source.zip"))) == 1
        git("add", "scripts")
        git("commit", "-qm", "Implementation")
        committed = capture_source(repo)
        assert committed["git_commit"] != original_head and committed["git_dirty"] is False
        assert dirty["git_commit"] == original_head  # Earlier experiment identity stays intact.
        source.unlink()
        deleted = capture_source(repo)
        with ZipFile(repo / deleted["source_archive"]) as archive:
            assert "scripts/example.py" not in archive.namelist()

        report = {"results": [{"model": "baseline", "variant": "reference"},
                              {"model": "baseline", "variant": "combined", "provenance": dirty}]}
        rows = "\n".join(provenance_rows(report, "Fixture"))
        assert "— (not recorded)" in rows and original_head in rows and "dirty" in rows
        assert committed["git_commit"] not in rows
        assert dirty["source_archive"] in rows
    print("Full commit IDs, dirty/untracked/staged source, archives and historical revisions passed.")


if __name__ == "__main__":
    main()
