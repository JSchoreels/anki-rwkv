# R01 GitHub Pages site

Static site (no build step). Pages:

- `index.html` (+ `home.js`): hub with links to the other pages and three interactive RWKV explainers (the five streams, the delta-rule state matrix, the forgetting-curve mixture). Toy data only.
- `models.html` (+ `models.js`): Alex (default) vs Expertium (his run 110 `s64hsm10`, rwkv-anki-autoresearch commit 1f2e6b0: d=64, 4 heads of 16, 3 learned power-law curves, no correction head; layer layout and the 333,178 parameter total are derived from this repository's w64 reconstruction plus 2 sibling-rating inputs, not his figures; this repository's older iter36 config is shown for reference) architecture: the 92 input features, the stream stacks, parameter waffles, per-layer split, recurrent state sizes and output heads. Counts are computed from configurations in `models.js` that mirror `rwkv/architecture.py`, `expertium/variants.py` (w64) and `expertium/architecture.py` (iter36); update them there if an architecture changes.
- `campaign.html` (+ `app.js`): the R01 GPU training-speed campaign. Charts use Chart.js and chartjs-plugin-annotation, loaded from jsDelivr at pinned versions.
- `train.html` (+ `train.js`): 8k/2k training-run progress.

The workflow also copies `docs/data-quality/` (review-relations results written by `scripts/publish_review_relations.py`) into the site at `/docs/data-quality/`, its original URL.

`style.css` holds the shared theme tokens; `explain.css` and `common.js` are shared by the hub and the models page.

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
