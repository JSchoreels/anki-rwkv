# HANDOFF: Faster low-rank codec in the fork (SM156)

Status 11-24 (night): **done, waiting for the user's OK to merge.** Result: `RESULT.md` here.

## Where things are

- Fork repo `C:/Users/jonat/workspace/anki`, branch `rc-lowrank-speed` (worktree
  `C:/Users/jonat/workspace/wt/fork-lowrank-speed`), one commit 47163a997 on top of `rc-2026-11` 514f3a41d, pushed.
  Changed: `rslib/src/rwkv/lowrank.rs`, new `rslib/src/rwkv/lowrank_reference.rs` (tests only: frozen copy of the
  merged codec), `docs/rwkv-lowrank-state-2026-11-23.MD` (speed paragraph).
- This branch `sm156` (anki-rwkv, worktree `C:/Users/jonat/workspace/wt/sm156`): `RESULT.md`, this file, timing logs
  and scripts.
- Not merged into `rc-2026-11`, fork `main` or any release branch; no tags, no releases, no GPU job.

## Outcome

- Bit-identical stored bytes and matrices vs the merged codec (new tests against the frozen copy; scorer parity
  counts unchanged: 252/252 at head size 32, 249/252 at head size 16; replay checksums identical).
- Codec write about 2 times faster (rank 4: 21.6 -> 11.0 µs at head size 16, 33.2 -> 16.0 µs at 32).
- Replay 1,000 cards x 24 answers, one thread, medians under load: Small T24 90.3 -> 72.4 µs per review, Tiny
  54.5 -> 39.2, Big 301.6 -> 237.0 (int8 rows 59.1, 31.05 and 252.3).
- Tests: `cargo test -p anki --lib` 1,564 passed / 46 ignored (default), 1,564 / 45 (no default features);
  clippy clean both ways; fmt clean.

## If the user approves the merge

    cd C:/Users/jonat/workspace/wt/fork-rc            # or any worktree on rc-2026-11 (check `git worktree list`)
    git fetch origin rc-2026-11:refs/remotes/origin/rc-2026-11
    git fetch origin rc-lowrank-speed
    git merge --ff-only origin/rc-lowrank-speed       # fast-forward if rc-2026-11 is still at 514f3a41d
    export PATH="/c/Users/jonat/.cargo/bin:$PATH" CARGO_TARGET_DIR=$PWD/out/rust
    cargo test -p anki --lib && cargo test -p anki --lib --no-default-features
    git push origin rc-2026-11                        # never force

If `rc-2026-11` moved, use a normal merge, re-run both suites on the merged tree, then push.

## Build setup (as in the low-rank states in the fork (SM151))

rustup MSVC toolchain first in PATH (`/c/Users/jonat/.cargo/bin`), `CARGO_TARGET_DIR=$PWD/out/rust`; the worktree's
untracked `out/` was copied from `wt/fork-lowrank` (`cp -a`), and `ftl/core-repo`, `ftl/qt-repo` contents without
`.git`.

## Follow-ups

- Clean re-time without a GPU job (`speed_run_alternating.sh`; it needs the frozen and current test binaries, rebuild
  them with `cargo test --release -p anki --lib --no-run` at 514f3a41d and at the branch tip).
- More speed needs more lockstep lanes than one write's four heads: batch the codec of several streams in the bulk
  replay (a bulk-sweep change).
- Optional: re-measure the Deck Options replay savings (four threads) with this codec.
