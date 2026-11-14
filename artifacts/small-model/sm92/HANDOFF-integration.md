# Show-time retrain of the release (SM92): fork integration and draft release, handoff (2026-11-04 ~08:15)

The user approved shipping the show-time retrain of the release (SM92) as the default small RWKV model in the Anki fork, with
commit, push and a **draft** release ("SM92, oui on integre dans anki, commit push et draft release"). **Never publish the
release and never mark it latest; the user publishes it.** Rules of the original brief still apply: queue every GPU job, cap
~0.8, quiet.json + daily 20:00-01:00 break (GPU and CPU), plain names with IDs in parentheses, never "Expertium" (say "outside
research"), FSRS7-Instant, commit as Jonathan Schoreels with the Co-Authored-By line, never bypass hooks, never comment on PRs
or issues, never stage `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`, `pages/data/factory-curated.json`,
`artifacts/coordination/queue/quiet.json`; never write to `C:\Users\jonat\AppData\Roaming\Anki2`. Fetch + ff first in every repo.

## Done

- **Package** `artifacts/models/sm92` (this worktree): `scripts/small_model_export.py package` (RWKV_DELETED_CARDS=neighbour,
  the sm72e25 options), parity PASS 1.1e-6 f32 / 1.2e-6 int8; `architecture.json` byte-identical to sm72e25's
  (`9bd41f11...`); weights.f32 `4fbc5f96...`, int8 `d17c619c...`, master `2091f8a8...`. Test vectors with **show-time query
  rows** (`testvectors/make_testvectors.py`, the scorer's `--inputs e2s-query` rows): committed rows identical to sm72e25's
  vectors, query rows differ only in inputs 2-7; NumPy checker PASS (2.4e-6). Manifest neutralised + evaluation
  (`package/sm92_package.py`), `INTEGRATION.md` with the new section 6.10 (show-time intervals). `master.pth` is in the folder
  but NOT committed (checkpoint).
- **Input convention check**: the fork already builds show-time inputs exactly as `show_time_intervals` training (query: now
  - card's last answer; commit: answer - previous answer; `FeatureState` cumulative sums committed intervals only). New Rust
  test `rwkv_sm92_feature_builder_matches_show_time_intervals`: all 6,543 queries + 7,860 answers of the full vectors match
  within 7.5e-7. No feature-builder change needed.
- **Fork WIP** on branch `sm92-default-model` (pushed; one WIP commit on top of `a40a4afc0`): bundle
  `qt/aqt/rwkv_inference/sm92/` (4 model files), default switch (`rwkv_model.py`, `profiles.py`, `lib.ts`,
  `RwkvModelOptions.svelte`), upgrade path (sm40/36/55/56/72/72e20/**72e25** -> sm92 on a Deck Options save), new key `sm92`
  (`ModelKind::Sm92`, leak-free rule), ftl, `.gitattributes`, `.dprint.json`, `build/configure/src/aqt.rs` (model list + the
  head copy loop for sm72e25 and sm92), tests (Python, TS, e2e, Rust), `qt/tools/rwkv_sm40_vectors.py --intervals`,
  fixture `rslib/src/rwkv/testdata/sm92-vectors.json.gz` (500 rows/user, with intervals), docs (`docs/rwkv.MD`,
  `docs/rwkv-sm92-2026-11-04.MD`, `RWKV_FAQ.md`, superseded note in `docs/rwkv-sm72e25-2026-10-30.MD`), `RELEASE.md`
  (build 110 notes frozen under `## 26.09.3+fsrs7.build.110 — 2026-11-01`; new `## Unreleased` "since build 110": SM92,
  the answer-time head, faster sync uploads).
- **Native parity** (harness `C:/Users/jonat/workspace/anki/rwkvharness`, `. ./env.sh; cargo test --offline --release`):
  full 14,403-row vectors 3.2e-6 (f32) / 3.0e-6 (INT8); fixture 2.0e-6 / 2.0e-6. Full harness run (default features):
  216 passed, 4 failed (the pre-existing `fused_packed` bit-exactness tests), 30 ignored, sm92 head tests skipped.

**Fork pushes:** a plain `git push` in the fork hung at 08:09 (the Git Credential Manager prompt is not visible from an
agent shell). Push with gh's credentials instead:
`GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin <branch>`.
The fork working tree was switched back to `main` (clean); the WIP is `origin/sm92-default-model` = `d06b3c7b3`.

## Placeholders still in the fork WIP (must all go before main)

- `rslib/src/rwkv/testdata/sm92-time-vectors.json.gz` = a COPY of `sm72e25-time-vectors.json.gz` (placeholder).
- `qt/aqt/rwkv_inference/sm92/` has no `time_head.f32.bin` / `time_head.int8.bin` / `time_head.json` yet.
- `docs/rwkv-sm92-2026-11-04.MD`: `⟨HEAD_TABLE⟩`, `⟨HEAD_HASHES⟩`, `⟨VERIFICATION⟩`.
- `RELEASE.md`: `⟨HEAD_VS_MEDIAN⟩`, `⟨HEAD_VS_LAST⟩`, `⟨HEAD_AE_VS_MEDIAN⟩`, `⟨HEAD_LL⟩`, `⟨HEAD_DELTA_MEDIAN⟩`, `⟨HEAD_AE⟩`,
  `⟨HEAD_PARITY⟩`, `⟨RWKV_COMMIT⟩` (the anki-rwkv sm92 commit that holds the packages).
- `git grep -n "⟨"` in the fork must print nothing before the final commit.

## Running: the answer-time head retrained on the frozen SM92 model (SM82c recipe)

Launcher `artifacts/small-model/sm92/head/launch-sm92-head.sh` (scripts of `wt/sm82c`, read only), WMI pid **7868**, started
07:45. Log `artifacts/small-model/sm92/head/gpu/launch-sm92-head.log`. CPU proof PASS, smoke OK (grade bitwise True, equal to
SM92's recorded scores True). Head training ticket running since 07:47 (`gpu/train.out`, ~200 steps/min, 9,999 steps ->
~08:40), then grade check (5 users), time scoring (723 users, ~15-20 min), then the gate (`head/sm92_head_gate.py`):
**PASS if time LogLoss <= 1.9568 x 1.01 = 1.9764 on the same 722-user evaluation and SM82c's rule passes**. Output
`head/sm92-head-decision.{md,json}` (`sm92_gate` block). Expected done ~09:05-09:15. Training folder
`artifacts/small-model/sm92-head` (`master.pth` = SM92 trunk + head; not committed).

```bash
cd C:/Users/jonat/workspace/wt/sm92 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm92/head/gpu/launch-sm92-head.log      # ends with "done" (or "STOP: ...")
tasklist //FI "PID eq 7868"
cat artifacts/small-model/sm92/head/sm92-head-decision.md | tail -3
```

Relaunch only if pid 7868 is gone and the log is not at `done` (resumes; finished steps are skipped):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/head/launch-sm92-head.sh >> C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/head/gpu/launch-sm92-head.nohup 2>&1"'; ProcessStartupInformation=$si}
```

**If the gate FAILS** (or the launcher STOPs and cannot be resumed): do not touch the fork's main; report to the caller
(gate numbers, `sm92-head-decision.md`), leave the WIP branch as is.

## Follow-up if the gate PASSES (one fresh agent; Sonnet is fine for steps 1-3, the fork part is mechanical)

`PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe`

1. **anki-rwkv, branch sm92** (this worktree): package the head (CPU, ~1-2 min; vectors on the same show-time rows as the
   model package, checked equal before writing):
   `$PY -X utf8 artifacts/small-model/sm92/package/sm92_time_package.py` -> `artifacts/models/sm92-time` (parity.json has
   the NumPy check). Commit with `git add -f` explicit paths: `artifacts/models/sm92/{INTEGRATION.md,architecture.json,
   manifest.json,parity.json,weights.f32.bin,weights.int8.bin}`, `artifacts/models/sm92/reference/*.py`,
   `artifacts/models/sm92/testvectors/{*.py,*.npz}`, `artifacts/models/sm92-time/{time_head.f32.bin,time_head.int8.bin,
   time_head.json,manifest.json,parity.json}`, `artifacts/models/sm92-time/reference/*.py`,
   `artifacts/models/sm92-time/testvectors/{*.py,*.npz}`, `artifacts/small-model/sm92/head/{sm92-head-decision.md,
   sm92-head-decision.json}`, `artifacts/small-model/sm92/head/gpu/{launch-sm92-head.log,launch-sm92-head.nohup,*.out,
   check-cpu-sm92.json,grade-check.json,smoke-grade-check.json,sm92-head-time.jsonl,sm92-head-time-ablation.jsonl}`,
   `artifacts/small-model/sm92-head/train.json`, this file. NOT: any `.pth` (master.pth, head.pth, resume.pt,
   ablation-head.pth), `gpu/grade-dump/`, `work/`. Push. Note the commit hash = `⟨RWKV_COMMIT⟩`.
2. **Fork** `C:/Users/jonat/workspace/anki`, branch `sm92-default-model` (fetch first; `git rebase origin/main` if main moved):
   - `cp <wt/sm92>/artifacts/models/sm92-time/time_head.{f32.bin,int8.bin,json} qt/aqt/rwkv_inference/sm92/`
   - `$PY qt/tools/rwkv_sm40_vectors.py <wt/sm92>/artifacts/models/sm92-time rslib/src/rwkv/testdata/sm92-time-vectors.json.gz --limit 500 --time-only`
   - full vectors (scratch, not committed): `$PY qt/tools/rwkv_sm40_vectors.py <wt/sm92>/artifacts/models/sm92-time <scratch>/sm92-time-full.json --limit 0 --intervals`
     and `$PY qt/tools/rwkv_sm40_vectors.py <wt/sm92>/artifacts/models/sm92 <scratch>/sm92-full.json --limit 0 --intervals`
   - harness: `cd rwkvharness && . ./env.sh && ANKI_RWKV_SM92_VECTORS=<scratch>/sm92-full.json ANKI_RWKV_SM92_TIME_VECTORS=<scratch>/sm92-time-full.json cargo test --offline --release sm92 -- --nocapture`
     (record the f32/INT8 bin-probability errors and the query count for the docs), then the fixture run without the env
     vars, then the whole suite `cargo test --offline --release` (expect only the 4 `fused_packed` failures) and
     `cargo test --offline --release --no-default-features` (same).
   - `rustfmt --edition 2021 --check rslib/src/rwkv/runtime.rs rslib/src/rwkv/sm40_tests.rs build/configure/src/aqt.rs`.
   - Fill every `⟨...⟩` (the decision's numbers: time LogLoss, vs user median and vs card's last time with CI and %, mean
     abs ln error; head file SHA-256s; parity; verification summary in the style of `docs/rwkv-sm72e25-2026-10-30.MD`
     "Verification"). In the RELEASE.md user-facing bullet, the percentages are relative LogLoss changes vs the baselines
     (SM72's head: 17.3% / 20.0% / 27.2%).
   - Squash to ONE commit on `main` (e.g. `git checkout main && git merge --squash sm92-default-model`), message:

```
Integrate the show-time retrain of the release (SM92) as the default small RWKV model

Bundle the new weights as sm92 next to the earlier small models and make
it the default: new profiles store sm92, unsupported values resolve to it,
and a successful Deck Options save upgrades SM40, SM36, SM55, SM56 and the
SM72 cool-downs (sm72, sm72e20 and sm72e25, all kept bundled) to it. A new
key keeps state cached with the earlier weights from being reused. Legacy
stays selectable.

SM92 is the SM72 cool-down from epoch 25 retrained on show-time intervals:
every query uses the time from the card's previous answer to the moment it
is shown, as Anki asks it, instead of the review log's answer-to-answer
interval. The fork already builds these inputs; a new test checks the
native feature builder against the trainer's interval inputs on every
query and answer of the release vectors. Same architecture and leak-free
input rule as SM72. Upstream, 722 users, both on show-time inputs: Instant
-0.000842 [-0.001136, -0.000630], Curve -0.000058 [-0.000271, +0.000255]
against SM72. Native parity on the full 14,403-row release vectors: 3.2e-6
(f32 weights) and 3.0e-6 (INT8 weights).

SM92 bundles its own answer-time head, retrained with the SM82c recipe on
the frozen SM92 model (time LogLoss <X> against 1.9568 for the SM72 head);
native parity <Y>.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

   - `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, push `main` (as 986963d23
     was). Delete the WIP branch afterwards (`git push origin --delete sm92-default-model`, `git branch -D ...`).
   - CI (`gh run list --repo JSchoreels/anki --workflow ci.yml --limit 3`) takes 15 min to 2 h: do not keep an agent
     waiting; hand off again if needed. Fix CI if it breaks (ruff/mypy/dprint/eslint were not runnable locally).
3. **Draft release** (documented path, RELEASE.md "Maintenance" + `.github/workflows/release.yml`): after CI is green on
   the main commit, run the unsigned draft build:
   `gh workflow run release.yml --repo JSchoreels/anki --ref main -f version=26.09.3+fsrs7 -f draft-release=true`
   (sign, testpypi, pypi default false). It resolves the tag `26.09.3+fsrs7.build.<run number>`, waits for CI itself,
   builds all platforms (~45-50 min) and creates the release with `--draft` from the `## Unreleased` section. Check with
   `gh release list --repo JSchoreels/anki --limit 3` (isDraft true) and `gh release view <tag> --repo JSchoreels/anki`.
   **Never** `gh release edit --draft=false`, never `--latest`.
4. Report: fork commit + CI, anki-rwkv commits, parity numbers, the head gate, the draft URL and tag, what is left for the
   user (publish the draft; optional title like build 107's).

## State 2026-11-04 ~09:55 (integration agent): fork main done, draft release building

- Head gate PASS 08:50 (time LogLoss 1.9580 vs 1.9568, +0.06%). Head packaged and committed on anki-rwkv `sm92`:
  `80035594ed0a3bf9577fcaff7cfa8d8b0aa2a122` (= `⟨RWKV_COMMIT⟩` in the fork's RELEASE.md link).
- Fork `main` (JSchoreels/anki):
  - `6caa5fb0e` Integrate the show-time retrain of the release (SM92) as the default small RWKV model (squash of
    `sm92-default-model`, plus the head files, the real `sm92-time-vectors.json.gz` fixture, filled notes/docs and a
    `use crate::rwkv::ModelKind;` fix in `sm40_tests.rs` for `--no-default-features`).
  - `056c7986b` Update pulldown-cmark to 0.13.6 for RUSTSEC-2026-0433 (new advisory failed cargo-deny on CI).
  - `38e641960` Release notes: pulldown-cmark 0.13.6 security update.
- Native parity (harness): model 3.2e-6 / 3.0e-6 (full), 2.0e-6 / 2.0e-6 (fixture); head 7.8e-7 / 7.6e-7 bin probability
  on 6,543 queries (fixture 407 queries: 7.2e-7 / 4.9e-7), history features exact; full suites 219 passed, 4 failed
  (the known `fused_packed`), 30 / 29 ignored.
- CI green on `38e641960`: https://github.com/JSchoreels/anki/actions/runs/41798626197 (attempt 2; attempt 1 failed
  only on the online link checker timing out on docs.ankiweb.net). `sm92-default-model` deleted (remote and local).
- **Release workflow running** (unsigned draft, run #111, started ~09:53):
  https://github.com/JSchoreels/anki/actions/runs/41802908390 . Expected tag `26.09.3+fsrs7.build.111`, notes from
  RELEASE.md `## Unreleased` (since build 110). ~45-50 min.

Follow-up (fresh agent, Sonnet is fine):

```bash
gh run view 41802908390 --repo JSchoreels/anki --json status,conclusion,jobs --jq '.status, .conclusion, (.jobs[] | [.name,.conclusion] | @tsv)'
gh release list --repo JSchoreels/anki --limit 3          # expect 26.09.3+fsrs7.build.111 with Draft
gh release view 26.09.3+fsrs7.build.111 --repo JSchoreels/anki --json isDraft,isPrerelease,url,body --jq '.isDraft, .isPrerelease, .url' 
gh release view 26.09.3+fsrs7.build.111 --repo JSchoreels/anki --json body --jq .body | head -20   # notes start "Changes since the last normal release, build 110"
```

If a build job fails: `gh run view 41802908390 --repo JSchoreels/anki --log-failed`; for a flaky/network failure
`gh run rerun 41802908390 --repo JSchoreels/anki --failed`. **Never** `gh release edit --draft=false`, never `--latest`;
the user publishes the draft (optional title in the style of build 107's).
