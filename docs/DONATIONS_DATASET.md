# Donations dataset (SM177)

`scripts/donations_to_dataset.py` turns the donated Anki collections into the anki-revlogs-10k layout, one user per donor, so that training and scoring can read them like the 10k dataset. It is needed before any model trained only on donations (MIT-licensable weights) can be trained. CPU only; nothing here touches the GPU.

## Output

`data/donations-YYYYMMDD/` (under `data/`, which is gitignored; never commit it):

| Path | Content |
|---|---|
| `revlogs/user_id=N/data.parquet` | `card_id`, `day_offset`, `rating`, `state`, `duration`, `elapsed_days`, `elapsed_seconds`, `review_time` (all int64) plus the pandas index column `__index_level_0__`: the schema of `data/anki-revlogs-10k` (the `-id` variant, builder flags `--raw-ids --show-time --emit-review-time`) |
| `cards/user_id=N/data.parquet` | `card_id`, `note_id`, `deck_id` (int64; deck = original deck for cards in a filtered deck) |
| `decks/user_id=N/data.parquet` | `deck_id`, `parent_id`, `preset_id` (int64; normal decks only, parent 0 when none) |
| `sources.parquet` | one row per user: `user_id`, `source` (campaign / LMSherlock / Expertium), `files` merged, `reviews_unique` (the inventory's count), `reviews` (rows kept), `cards_reviewed`, `reviews_deleted_cards`, `cards`, `decks`, and the day boundary used (`day_rule`, `rollover_hour`, `rollover_source`, `utc_offset_minutes_west`, `offset_source`) |
| `summary.json` | aggregate counts, the count check and the loader check |

- **User ids** are 1,000,001, 1,000,002, ... in the inventory's donor-group order, so they cannot collide with the 10k dataset (1..10,000). They are per snapshot: a rerun with new files can renumber donors.
- **Entity ids** are the raw Anki ids, as in the `-id` variant. The model only uses them as keys (`docs/SMALL_MODEL_INTEGRATION.md` 6.1).
- **Answer-time layout** (`--answer-time-out DIR`): the same users in the layout of `data/anki-revlogs-10k-original-20261011` (no `review_time`; `day_offset` and the intervals from the answer time, i.e. the builder without `--show-time`). The training run (`scripts/run_full_training.py`) and the held-out scorer read that layout; the show-time queries come from `--inputs e2s-query` there.

## Merge rules

1. **Files** are read with `scripts/donor_inventory.py`'s own functions (`iter_sources`, `read_collection`): `.apkg`, `.colpkg`, `.anki2`/`.anki21` and `.zip` archives of packages (zstd via Git's `libzstd.dll`). Other files (for example a `.revlog` research export) are ignored, by the inventory too.
2. **Donors** are the inventory's groups (`donor_inventory.group_donors`, the same code its report uses): files that share 10%+ of their review ids, review in the same 10-minute windows at least half of the time, or come from one archive belong to one donor. Exact duplicates fall into their original's group.
3. **Reviews**: a donor's reviews are the union of the answered review ids (ease 1-4) of its files, each review once. When several files hold the same review id, the row comes from the first file (in priority order) that still has the card, else from the first file. Priority order = the file with most answered reviews first, then file order.
4. **Cards and decks** come from the first file in priority order that holds them. The cards table holds every card of the donor's files, the decks table every normal deck, as the research export does.
5. **Deleted cards**: reviews whose card is in none of the donor's files are kept and the card stays out of `cards`, as in the 10k dataset. The loader then gives such a card its own note (`314159265358979323 + card id`), the shared placeholder deck and preset, and the flags 13-15.
6. **Dataset-builder rules** (anki-revlogs-dataset-builder `e299465`, `process_revlogs`; the export side is rslib `export_dataset`):
   - kept: `ease` 1-4 (the export's `button_chosen >= 1`), every review type 0-5, except filtered-deck rows without rescheduling (type 3 with factor 0);
   - per card, in (card id, review id) order: only the rows from the card's last Learning start (a type-0 row whose previous row of the card is not type 0); a card without a Learning start is dropped entirely;
   - `state` = review type + 1, 0 on that Learning start;
   - `duration` = the revlog `time` in ms as rslib decodes it (a value that is not an integer in u32 range becomes 0), no cap;
   - `review_time` = review id - duration (show time); `day_offset` = trunc((review_time / 1000 - next_day_at) / 86400), shifted so the user's first day is 0;
   - `elapsed_days` = `day_offset` difference and `elapsed_seconds` = trunc(review_time difference / 1000), clamped at 0 (seconds only), from the card's previous kept row in (card id, review id) order; -1 on state 0;
   - rows sorted by `review_time` (stable sort; ties keep the card/review order; the builder's sort has no tie-break).
7. **Counts**: `reviews_unique` is the inventory's unique count (`donor_inventory.merge_group`); `reviews` is smaller because of the builder rules in 6 (cram rows, history before the last Learning start, cards never in Learning).

## Day boundary

The 10k export takes `next_day_at` from Anki's `timing_today` on the exporting machine: the next rollover at the rollover hour in the machine's current UTC offset, with fixed 86,400 s days (no daylight-saving shifts). The converter rebuilds it from each donor's collection config (the top-priority file that has one):

- **rollover hour**: config `rollover` (modern `config` table, or legacy `col.conf`); missing = 4, the value Anki itself writes when a v2 collection lacks it (`rollover_source = default-4`);
- **UTC offset**: config `localOffset` (minutes west of UTC, written by the Anki client each time the collection is opened, so the donor's last-used timezone); if missing `creationOffset`; if both are missing **UTC** (`offset_source = fallback-utc`): the days then roll over at the rollover hour UTC;
- **v1 scheduler** (`schedVer` 1): days start at the collection creation time `crt` + k days, as Anki's v1 timing does (`day_rule = v1-crt`);
- `next_day_at` = the first such boundary after the donor's last review. Only its phase matters, since `day_offset` is shifted to start at 0.

`summary.json` (`day_boundary`) counts the users per rule and fallback.

## First build (2026-11-26)

`data/donations-20261126` (and `-answer-time`), from the 264 collections of the three folders (0 read errors), 321 s, peak 1.1 GB:

| Source | Users | Unique reviews (inventory) | Reviews kept |
|---|---:|---:|---:|
| campaign | 51 | 13,505,432 | 12,954,035 |
| LMSherlock | 107 | 19,071,378 | 18,096,822 |
| Expertium | 60 | 9,375,952 | 8,840,564 |
| **total** | **218** | **41,952,762** | **39,891,421** |

- 224 donor groups; 6 have no answered review and get no user. 218 users, ids 1,000,001 to 1,000,218.
- **Count check**: the converter's union count equals `merge_group`'s for all 218 donors (0 mismatches), and the totals per source equal the published inventory aggregate of the same files (`pages/data/donations.json`, 19:49).
- **Loader check**: 5 users (221 to 253,125 rows) build their features with `get_rwkv_data` in both layouts; the only non-finite column is `label_review_th` on each card's last row, as for the 10k dataset.
- **Day boundary**: rollover from the config for 176 users, default 4 for 42; offset from `localOffset` 176, `creationOffset` 38, UTC fallback 4; no v1 collection.
- **Smoke test** (information only): the six-layer Tiny two-head model (SM130), w8+s8, on 20 users spread over sizes up to 400k reviews (1,621,900 benchmark test reviews), CPU, the 732-set protocol (answer-time layout, `--inputs e2s-query`, `--deleted-cards neighbour`, FSRS-6 benchmark test reviews built with `rwkv/find_equalize_test_reviews.process` on the converted users): review-weighted LogLoss curve 0.3345 and instant 0.3237, against 0.3014 and 0.2780 on the 732 set. Per-user median curve 0.344 vs 0.307. Different users, so this is no accuracy comparison; the run only shows that the converted data scores end to end without errors. About 2,900 review-evaluations/s on 2 threads.

## Privacy

- The donation folders are opened read-only. Packages are extracted, read and deleted one at a time in a temp folder outside the repository; the per-file arrays live there too and the whole folder is deleted at the end (`--keep-scratch` keeps it, for debugging only).
- The output holds no donor name, file name, deck name or note text: ids, ratings, times and the source label only. `sources.parquet` and `summary.json` are local data like the rest of `data/`.
- Source labels (campaign / LMSherlock / Expertium) are the folder a donor's reviews mostly came from; a donor is never labelled by name.
- Never commit `data/donations-*`; published pages show aggregates only (`scripts/publish_donations_page.py`). Never read the live Anki profile folder (`%APPDATA%\Anki2`).

## Rerun

```
OMP_NUM_THREADS=4 .venv/Scripts/python -X utf8 scripts/donations_to_dataset.py \
    --out data/donations-YYYYMMDD --answer-time-out data/donations-YYYYMMDD-answer-time --validate 5
```

`--folder` (repeatable) and `--source-labels` replace the three default folders and their labels. The output folders must be new or empty. The process runs at below-normal priority with at most 4 threads per library; memory stays well below 6 GB (one collection at a time in pass 1, one donor at a time in pass 2; review ids as numpy arrays).

Validation built in: the per-donor union count must equal `donor_inventory.merge_group`'s count (`count_check`), and `--validate N` builds the training features of N users (spread over sizes) with `rwkv.data_processing.get_rwkv_data` (`loader_check`).
