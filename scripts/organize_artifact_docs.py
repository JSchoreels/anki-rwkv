"""Relocate textual artifact references and keep a concise human-facing index."""
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "artifacts/layout.json").read_text(encoding="utf-8"))
mapping = sorted(manifest["mappings"].items(), key=lambda item: len(item[0]), reverse=True)

def relocate(path):
    absolute = Path(os.path.abspath(path))
    for old, new in mapping:
        old_root = ROOT / old
        try:
            suffix = absolute.relative_to(old_root)
        except ValueError:
            continue
        return ROOT / new / suffix
    return absolute

def original_location(path):
    for old, new in mapping:
        try:
            return ROOT / old / path.relative_to(ROOT / new)
        except ValueError:
            continue
    return path

def rewrite_links(text, path):
    old_parent = original_location(path).parent
    def replace(match):
        href = match.group(1)
        bracketed = href.startswith("<") and href.endswith(">")
        target = href[1:-1] if bracketed else href
        if not target or target.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            return match.group(0)
        filename, separator, fragment = target.partition("#")
        destination = relocate(old_parent / filename)
        relative = os.path.relpath(destination, path.parent).replace(os.sep, "/")
        rewritten = relative + (separator + fragment if separator else "")
        if bracketed:
            rewritten = "<" + rewritten + ">"
        return "](" + rewritten + ")"
    return re.sub(r"\]\(([^)\n]+)\)", replace, text)

def rewrite_prefixes(text):
    for old, new in mapping:
        for separator in ("/", "\\\\", "\\"):
            old_variant = old.replace("/", separator)
            new_variant = new.replace("/", separator)
            pattern = re.escape(old_variant) + r"(?=$|[/\\\s\"'`),.;:#])"
            text = re.sub(pattern, lambda match: new_variant, text)
    return text

ledger = ROOT / "docs/GPU_OPTIMIZATIONS.md"
history = ROOT / "docs/GPU_OPTIMIZATION_HISTORY.md"
finish_only = sys.argv[1:] == ["--finish-ledger"]
if history.exists() and not finish_only:
    raise FileExistsError("Existing history must be preserved; do not overwrite it")
if not finish_only:
    history.write_bytes(ledger.read_bytes())

files = {ROOT / "README.md", ROOT / "justfile"}
for name in ("scripts", "configs", "docs", "rwkv", "expertium"):
    for path in (ROOT / name).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".ps1", ".toml", ".md", ".html"}:
            if "implementations" in path.relative_to(ROOT).parts:
                continue  # Preserved historical source copies are not live runners.
            files.add(path)
for group in ("benchmarks", "diagnostics", "retraining", "data"):
    files.update((ROOT / "artifacts" / group).rglob("*.md"))
changed = []
for path in ([] if finish_only else sorted(files)):
    before = path.read_text(encoding="utf-8")
    after = rewrite_links(before, path) if path.suffix.lower() == ".md" else before
    after = rewrite_prefixes(after)
    if after != before:
        path.write_text(after, encoding="utf-8")
        changed.append(path.relative_to(ROOT).as_posix())

full = ledger.read_text(encoding="utf-8")
track_section = full[full.index("| ID | Track | Priority / effort |"):full.index("\n\nF14 is distinct")]
track_lines = [line for line in track_section.splitlines() if re.match(r"^\| F\d+ \|", line)]
if len(track_lines) != 13:
    raise ValueError(f"Expected 13 F tracks, found {len(track_lines)}")
r01_start = full.index("### R01:")
r01_end = full.index("\nF13 tests ", r01_start)
r01 = full[r01_start:r01_end].strip()
bundle = next(line for line in full.splitlines() if line.startswith("| B01 |"))
bundle_cells = bundle.split("|")
bundle_cells[3] = " Original controls disagree; deterministic retry incomplete; acceptance pending "
bundle_cells[6] = " Establish matched-control repeatability before assessing the fixed bundle; see the latest run and iteration record. "
bundle = "|".join(bundle_cells)
commits = full[full.index("| Change / bundle |"):full.index("### Validation gate and agreed user count")].strip()
lines = [
"# GPU optimization ledger", "",
"Start with the current reports below. This file keeps the track list and decisions; detailed measurements and diagnostics live with their runs.", "",
"## What to read", "",
"- [Artifact guide](../artifacts/README.md): where results, diagnostics and caches are stored.",
"- [B01 comparison](../artifacts/benchmarks/gpu-baseline-b01/summary.md): original controls disagreed; no accepted bundle result.",
"- [Deterministic retry](../artifacts/benchmarks/gpu-baseline-b01-deterministic-v3/summary.md): partial attempt preserved; see iteration 0001 for the next action.",
"- [F01–F08 results](../artifacts/benchmarks/gpu-batch1-500/summary-relative.md): five completed cases from the stopped first batch.",
"- [R01 procedure](GPU_RETRAIN_LOOP.md): the reusable optimization loop.",
"- [Full historical notes](GPU_OPTIMIZATION_HISTORY.md): previous tables, protocols, checks and investigations, preserved for reference.", "",
"## How to interpret results", "",
"- Report Curve and Instant binary logloss separately, both label-weighted and user-averaged.",
"- Compare every candidate with its own matched control; report training time, raw review presentations/s, exposure and timing scope.",
"- Repeated-window throughput is a preliminary screen. Expertium's 99,099 reviews/s result does not establish full-history speed or quality.",
"- B01 keeps the existing absolute 0.001 quality and repeated-control stability gates. R01 later uses a fixed total <1% relative regression budget.",
"- Preserve raw results, failed/inconclusive attempts and process-start source provenance. A completed run is not automatically validated.", "",
"<!-- optimization-tracks:start -->", "## Optimization tracks", "",
"Statuses and measurements below are retained from the existing ledger. Missing values remain `—`; the linked run reports provide the full context.", "",
"| ID | Track | Priority / effort | Status | Curve / Instant log loss | Measured s / reviews/s | Next experiment |",
"|---|---|---|---|---|---|---|",
*track_lines,
"", "F12's data-quality investigation is tracked as DQ01 in [DATA_QUALITY.md](DATA_QUALITY.md).",
"F03 packs prepared CPU tensors; F14 materializes packed feature metadata on the GPU.",
"<!-- optimization-tracks:end -->", "",
"## B01: execution baseline", "",
"B01 combines F14/F01/F03/F04/F05 exact shapes/F06, with no intended change to model or training behavior. Bundle acceptance does not independently validate each component.", "",
"| Bundle ID | Description | Status | Full-history Curve / Instant log loss | Complete training s / reviews/s | Next step |",
"|---|---|---|---|---|---|", bundle, "",
r01, "", "## Validation and provenance", "",
"- Broader validation requires an explicit user request and at least 500 distinct evaluation users sampled from IDs 1–5000.",
"- 100/200-user R01 stages are preliminary. Preserve chronological holdouts, paired-user uncertainty and the reserved independent final test.",
"- Record configurations, exact cohort/seed/budget rules, precision, updates and acceptance criteria before each comparison.",
"- Preserve the experiment commit, dirty state and exact source snapshot; record implementation commits separately.",
"- Follow [AGENT.md](../AGENT.md) for the complete repository requirements.", "",
"## Implementation commits", "", commits, "",
]
ledger.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"rewritten_files": changed, "track_count": len(track_lines), "ledger_lines": len(lines), "history": history.relative_to(ROOT).as_posix()}, indent=2))
