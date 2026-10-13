# R01 GitHub Pages site

Static site (no build step) visualising the R01 GPU training-speed campaign. Charts use Chart.js and
chartjs-plugin-annotation, loaded from jsDelivr at pinned versions.

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

then open http://localhost:8000/ (the page fetches `data/r01.json`, so `file://` will not work).

## Publish

Enable Pages once in the repository settings: Settings > Pages > Source: **GitHub Actions**.
After that `.github/workflows/pages.yml` deploys the `pages/` folder on every push to `main` that
touches `pages/**`, or manually via the workflow_dispatch button.
