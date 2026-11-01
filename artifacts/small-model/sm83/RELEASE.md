# Answer-time head release (SM83): release steps and the publish commands

Prepared Sunday 2026-11-01 for the answer-time head on richer frozen features (SM82c) with the released small model (the leak-free small model, cool-down from epoch 25 (SM72), fork key `sm72e25`). The user's go-ahead: "yeah let's ship the time head" (11-01 ~10:00).

## Done (no confirmation needed: the user's own repos)

| step (as the 10-30 release) | where | state |
|---|---|---|
| release package (export, NumPy reference, vectors, parity, manifest, manual section, zip, release notes) | anki-rwkv branch `sm83`: `artifacts/models/sm72e25-time/`, `sm72e25-time-package.zip` (7,817,995 B, SHA-256 `6dcd8fca0cfb5a7ff5075343b9e538ba608772a9901fdf4b15c423c1e0b8f04d`), `sm72e25-time-release-notes.md` | committed, pushed |
| fork integration: head file bundled byte-identical (`time_head.f32.bin` SHA-256 `e958d749…258d223b`), runtime, RPC, bridge, Card Info, tests, fixture, `RELEASE.md` Unreleased entry, `docs/rwkv-answer-time-2026-11-01.MD`, `docs/rwkv.MD` | `JSchoreels/anki` branch `rwkv-answer-time-head`, commit 826df061e, PR #24 | pushed; **CI pending** |

Difference from 10-30: that integration was committed straight to `main`; this one touches the protobuf API, the PyO3 bridge and desktop Python that cannot be built or run on this machine (no protoc / ninja / C compiler / PyQt / pytest), so it goes through PR #24 to get the full CI first. The model key stays `sm72e25` (no new key, no migration): the model weights are byte-identical, so cached state is valid.

## Remaining steps

1. **Merge PR #24 once CI is green** (normal work, the user's repo; previous fork PRs were merged with merge commits):

   ```
   gh pr checks 24 --repo JSchoreels/anki --watch
   gh pr merge 24 --repo JSchoreels/anki --merge
   ```

   If CI fails: fix on `rwkv-answer-time-head` (fetch + ff first), push, re-check. The local harness covers `rslib/src/rwkv/` only (see the fork's integration report, "Verification").

2. **Needs the user's confirmation (publishing): a fork build.** The fork's builds are published by the release workflow (10-31 23:50: `Release 26.09.3+fsrs7 (sign=false, draft=true, ...)` on `main`, then the user published the draft as pre-release `26.09.3+fsrs7.build.109`). `.version` stays `26.09.3+fsrs7`; `RELEASE.md`'s Unreleased section already holds the entry (renamed only for a normal release). After the merge:

   ```
   gh workflow run release.yml --repo JSchoreels/anki --ref main -f version=26.09.3+fsrs7 -f draft-release=true
   gh run list --repo JSchoreels/anki --workflow release.yml --limit 1
   # then, in the GitHub UI or: gh release edit <draft tag> --repo JSchoreels/anki --draft=false --prerelease
   ```

3. **Optional, needs the user's confirmation (publishing): a GitHub release of the package in anki-rwkv**, as for the 10-28 / 10-29 small-model releases (the 10-30 package had none). After `sm83` is merged into `main` (or tag the `sm83` commit):

   ```
   gh release create small-model-sm72e25-time-2026-11-01 C:/Users/jonat/workspace/anki-rwkv/artifacts/models/sm72e25-time-package.zip \
     --repo JSchoreels/anki-rwkv --target <commit with the package on main> \
     --title "Answer-time head for the small model, leak-free, cool-down from epoch 25 (SM72): 69k parameters, -17% time LogLoss vs the running median, model unchanged" \
     --notes-file C:/Users/jonat/workspace/anki-rwkv/artifacts/models/sm72e25-time-release-notes.md
   ```

Nothing was published outside the user's own branches and PR.
