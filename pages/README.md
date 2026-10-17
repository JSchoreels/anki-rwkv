# R01 GitHub Pages site

Static site (no build step). Pages:

- `index.html` (+ `home.js`): hub with links to the other pages and three interactive RWKV explainers (the five streams, the delta-rule state matrix, the forgetting-curve mixture). Toy data only.
- `models.html` (+ `models.js`): Alex (default) vs Expertium (his run 110 `s64hsm10`, rwkv-anki-autoresearch commit 1f2e6b0: d=64, 4 heads of 16, 3 learned power-law curves, no correction head; layer layout and the 333,178 parameter total are derived from this repository's w64 reconstruction plus 2 sibling-rating inputs, not his figures; this repository's older iter36 config is shown for reference) architecture: the 92 input features, the stream stacks, parameter waffles, per-layer split, recurrent state sizes and output heads. Counts are computed from configurations in `models.js` that mirror `rwkv/architecture.py`, `expertium/variants.py` (w64) and `expertium/architecture.py` (iter36); update them there if an architecture changes.
- `campaign.html` (+ `app.js`): the R01 GPU training-speed campaign. Charts use Chart.js and chartjs-plugin-annotation, loaded from jsDelivr at pinned versions.
- `train.html` (+ `train.js`): 8k/2k training-run progress.
- `factory.html` (self-contained, fetches `data/factory.json`): factory overview of the whole work (today one track, "small model"): 1. what runs now (with progress), 2. the planned timetable with live status, 3. the backlog with a status filter, 4. every GPU and CPU result with absolute LogLoss, the gap to the fork and the evaluation size. `small-model.html` is a redirect to it.

The workflow also copies `docs/data-quality/` (review-relations results written by `scripts/publish_review_relations.py`) into the site at `/docs/data-quality/`, its original URL.

`style.css` holds the shared theme tokens; `explain.css` and `common.js` are shared by the hub and the models page.

## Factory overview data and its publisher

- `data/factory-curated.json` is edited by hand: headline cards (`k`, `v`, then either `s` as one line of text or `compare`: a list of rows {`variant`, `curve`, `instant`, `curve_pct`, `instant_pct`, `eval`, `ref`} drawn as an aligned table, where the `ref: true` row is the fork and a missing % is computed against it; optional `link` {`href`, `text`}), `now_notes` (text for a running ticket, found by a `match` substring of its queue purpose; `manual: true` entries are CPU jobs outside the queue); a running ticket without a `now_notes` match takes its What / Decides / ID from the plan row whose `match` it hits, and finished / waiting tickets show that plan row's ID, `plan` rows (optional `match`: a substring, or a list of substrings, of the queue ticket purpose; the generator then emits the state plus actual `started_at` / `ended_at` (done) or `started_at` / `ends_at` (running); `minutes` = the expected duration; a row without `match` and `minutes` keeps a fixed `when`). The browser lays the timetable out at page load: done rows (older than 6 h collapse into one line), the running row, then pending rows chained from the running job's end, each `minutes` long. `gpu_break` {start, end, until, label} is the GPU break: a pending job that would end after the break starts begins when it ends, and the break is shown as its own row; it is ignored after `until` (delete the key to remove it). Times are not in the material hash, so the loop commits only on state changes, `backlog`, `results` (takeaways live here) and `known_labels` (scoring labels deliberately not shown as results). Every row has a `track` ("small model"); add rows for other workstreams with their own track.
- `data/factory.json` is generated, never edited: `.venv/Scripts/python.exe scripts/publish_factory_page.py [--check] [--force]`. It reads the curated file, `scripts/r01_queue.py status` and the queue ledger (`artifacts/coordination/queue/history.jsonl`), the newest `runbook-*.log` of the `small-model` worktree (progress, reviews/s, ETA) and its `heldout.jsonl`: every `*/w8+s8` label with 464 users that is not in the curated results or `known_labels` becomes a result row with review-weighted absolute LogLoss for both heads, the gap to `fork/w8+s8`, the evaluation size and the takeaway "pending". The flow audit (Haiku) replaces "pending" by writing the takeaway into the curated row (add `"label"` to the row so it is not added twice). `--check` prints what would change; the file is rewritten only when a ticket started, finished or was queued, a plan state or result label changed, or the curated file changed (the `material` hash), or with `--force`. Exit code 3 = nothing material.
- `scripts/publish_factory_loop.sh` runs the publisher every 5 minutes, commits `pages/data/factory.json` alone ("Pages: factory status ...") and pushes, which deploys the Page. Start: `nohup bash scripts/publish_factory_loop.sh > /dev/null 2>&1 &`. Stop: `kill "$(cat artifacts/factory-page/loop.pid)"`. Log: `artifacts/factory-page/loop.log`. It skips a cycle when `.git/index.lock` exists, when something is staged, or when the checkout is not on `main`.

## Regenerate the data

From the repository root (CPU only, stdlib only, read-only on the repo apart from the output file):

```
.venv/Scripts/python.exe scripts/build_r01_pages_data.py
```

This rewrites `pages/data/r01.json` from `docs/GPU_OPTIMIZATIONS_LOG.md`, the per-iteration
`comparison.json` / `result.json` files and the ledger in `docs/GPU_OPTIMIZATIONS.md`. Rerun it whenever
new iterations land, then commit `pages/data/r01.json`. New milestones are curated in the `MILESTONES`
list inside the script.

## Preview locally

```
cd pages && python -m http.server 8000
```

then open http://localhost:8000/ (pages fetch `data/*.json`, so `file://` will not work).

## Publish

Enable Pages once in the repository settings: Settings > Pages > Source: **GitHub Actions**.
After that `.github/workflows/pages.yml` deploys the `pages/` folder on every push to `main` that
touches `pages/**`, or manually via the workflow_dispatch button.
