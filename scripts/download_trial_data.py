"""Download all three dataset tables for a selected inclusive user range."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "huggingface"))

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--user-start", type=int, default=1)
    parser.add_argument("--user-end", type=int, default=100)
    parser.add_argument("--repository", default="open-spaced-repetition/anki-revlogs-10k")
    parser.add_argument("--output", default=str(ROOT / "data" / "anki-revlogs-10k"))
    args = parser.parse_args()
    if not 1 <= args.user_start <= args.user_end:
        raise ValueError("Invalid user range")
    repo = args.repository
    manifest_path = Path(args.output) / "download_manifest.json"
    if manifest_path.is_file():
        existing_repo = json.loads(manifest_path.read_text(encoding="utf-8")).get("repository")
        if existing_repo != repo:
            raise ValueError(
                f"Output already contains {existing_repo}; select that repository or a different output directory")
    api = HfApi()
    revision = api.dataset_info(repo).sha
    selected = []
    coverage = set()
    for name in api.list_repo_files(repo, repo_type="dataset", revision=revision):
        match = re.fullmatch(r"(cards|decks|revlogs)/user_id=(\d+)/[^/]+\.parquet", name)
        if match and args.user_start <= int(match[2]) <= args.user_end:
            selected.append(name)
            coverage.add((match[1], int(match[2])))
    expected = {(table, user) for table in ("cards", "decks", "revlogs")
                for user in range(args.user_start, args.user_end + 1)}
    missing_metadata = sorted(expected - coverage)
    missing_reviews = [item for item in missing_metadata if item[0] == "revlogs"]
    if missing_reviews:
        raise RuntimeError(f"Missing review histories in source dataset: {missing_reviews}")
    # Some collections have no cards/decks metadata. The existing preprocessing
    # intentionally fills missing hierarchy IDs after its left joins.
    if missing_metadata:
        print(f"Source has absent metadata partitions (handled by preprocessing): {missing_metadata}", flush=True)
    print(f"Downloading {len(selected)} files for users {args.user_start}-{args.user_end} at {revision}", flush=True)
    selected.extend(["README.md", "LICENSE"])
    output = Path(snapshot_download(repo, repo_type="dataset", revision=revision,
                      allow_patterns=selected, local_dir=args.output, max_workers=8))
    absent = [name for name in selected if not (output / name).is_file()]
    if absent:
        raise RuntimeError(f"Download incomplete: {absent}")
    manifest = {"repository": repo, "revision": revision, "user_start": args.user_start,
                "user_end": args.user_end, "files": selected,
                "bytes": sum((output / name).stat().st_size for name in selected),
                "missing_metadata_partitions": missing_metadata}
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {len(selected)} files, {manifest['bytes'] / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
