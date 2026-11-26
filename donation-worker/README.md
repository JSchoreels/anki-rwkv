# Anki-RWKV campaign uploads

GitHub Pages serves the public campaign and its totals. This Worker issues a 15-minute Dropbox upload link for a new, randomly named collection under `/Anki Data`. Anki posts the archive directly to Dropbox. The Worker accepts an empty POST at `/v1/upload-link` and returns JSON with a `link` field on success. Anki checks Dropbox’s receipt before scheduling the next donation.

Run `node --test worker.test.mjs` to check the route. Deploy this directory with Wrangler 4.36 or later using `wrangler deploy`. The configuration binds the per-location rate limiter used for link requests.

Register a scoped Dropbox app with Full Dropbox access to reach the existing root folder. Enable file-content write and authorize the campaign owner with PKCE and offline access. Set `DROPBOX_APP_KEY` and `DROPBOX_REFRESH_TOKEN` in the Worker’s encrypted secrets. The refresh token must come from a PKCE grant for this app. Removing the refresh token disables new upload links. Keep the refresh token out of the repository, Pages and the Anki client.
