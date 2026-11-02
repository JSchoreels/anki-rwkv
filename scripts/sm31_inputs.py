"""Input-realism variants for the held-out scorer (scripts/small_model_heldout.py --inputs; grade heads only).

duration-leak scoring (SM31): the published revlogs' `elapsed_seconds` runs from the card's previous answer to this
answer, so a review's interval inputs contain the review's own answer time (`duration`), which is unknown when Anki
shows the card (docs/EXPERTIUM_CATCH_UP.md section 4; response-time prediction (SM82) stage-0 "Leak audit").
  e2s-query  show-time intervals, the deployment-realistic inputs for a model trained on answer-to-answer intervals:
             only what is asked before the answer moves. The instant query row of review j gets the interval from the
             card's previous answer to the moment j is shown, floor(max(elapsed_seconds - duration / 1000, 0)), with
             its sin / cos and the cumulative interval (the card's earlier answer-to-answer intervals + that show-time
             interval) and their sin / cos; the curve query for review j (label_elapsed_seconds of the card's previous
             row) asks at that same show-time interval. Update rows (after the answer, when the full interval and the
             answer time are known) keep their inputs.
  e2s-all    the outside-research definition (expertium/intervals.end_to_start_published, EXPERTIUM_END_TO_START=1):
             every row's elapsed_seconds becomes end-to-start, so the update rows and the cumulative interval move too.
robustness to other apps (SM29): inputs another app could give us; labels are never touched.
  passfail   ratings collapsed to pass / fail in the inputs: Hard, Good and Easy rows get the Good one-hot
             (rating_1..rating_4 columns only; the `rating` column feeds the instant labels and stays).
  notime     no answer times: scaled_duration as for a missing time (duration 0) on every row.
  passfail-notime  both.
`normal` = today's frame and samples, untouched.
"""
import contextlib
import os

import numpy as np

VARIANTS = ("normal", "e2s-query", "e2s-all", "passfail", "notime", "passfail-notime")
SECONDS_PER_DAY = 86400
QUERY_INTERVAL_COLUMNS = ("elapsed_seconds", "elapsed_seconds_sin", "elapsed_seconds_cos", "elapsed_seconds_cumulative",
                          "elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos", "scaled_elapsed_seconds",
                          "scaled_elapsed_seconds_cumulative")


def show_time_seconds(seconds, duration_ms):
    """expertium/intervals.end_to_start_published on arrays: floor(max(s - d / 1000, 0)), the -1 sentinel kept."""
    seconds = np.asarray(seconds, dtype="float64")
    out = np.floor(np.maximum(seconds - np.asarray(duration_ms, dtype="float64") / 1000.0, 0.0))
    out[seconds == -1] = -1.0
    return out


def load_frame(data, user, variant):
    """get_rwkv_data(data, user) with the variant's frame-level change (before any deleted-card transform)."""
    from rwkv import data_processing as dp
    if variant not in VARIANTS:
        raise ValueError(f"unknown input variant {variant}")
    if variant == "e2s-all":
        old = os.environ.get("EXPERTIUM_END_TO_START")
        os.environ["EXPERTIUM_END_TO_START"] = "1"
        try:
            return dp.get_rwkv_data(data, user)
        finally:
            if old is None:
                os.environ.pop("EXPERTIUM_END_TO_START", None)
            else:
                os.environ["EXPERTIUM_END_TO_START"] = old
    if os.environ.get("EXPERTIUM_END_TO_START") == "1":
        raise SystemExit("--inputs: EXPERTIUM_END_TO_START must be unset (the variants define the intervals)")
    return dp.get_rwkv_data(data, user)


def apply_frame(frame, variant):
    """Variant columns on the finished frame (after the deleted-card transform); `normal` / `e2s-all` return it as is."""
    from rwkv import data_processing as dp
    if variant in ("normal", "e2s-all"):
        return frame
    frame = frame.copy()
    if variant == "e2s-query":
        show = show_time_seconds(frame["elapsed_seconds"].to_numpy(), frame["duration"].to_numpy())
        nxt = frame.assign(_show=show).groupby("card_id")["_show"].shift(-1)
        frame["label_elapsed_seconds"] = nxt.fillna(np.pi).astype(frame["label_elapsed_seconds"].dtype)
        return frame
    if variant in ("passfail", "passfail-notime"):
        passed = frame["rating"].to_numpy() > 1
        for k in (2, 4):
            frame[f"rating_{k}"] = np.where(passed, 0, frame[f"rating_{k}"]).astype(frame[f"rating_{k}"].dtype)
        frame["rating_3"] = np.where(passed, 1, frame["rating_3"]).astype(frame["rating_3"].dtype)
    if variant in ("notime", "passfail-notime"):
        frame["scaled_duration"] = np.full(len(frame), dp.scale_duration(0.0), dtype=frame["scaled_duration"].dtype)
    return frame


def _query_rows_show_time(section_df):
    """e2s-query: the add_queries frame with the instant query rows' interval columns at show time."""
    from rwkv import data_processing as dp
    q = section_df["is_query"].to_numpy() == 1
    if not q.any():
        return section_df
    sub = section_df.loc[q]
    # query rows carry duration 0 (rejected column): the answer time comes from the review's own update row
    own = section_df.loc[~q, ["review_th", "duration"]].set_index("review_th")["duration"]
    duration = own.reindex(sub["review_th"].to_numpy()).to_numpy()
    if np.isnan(duration.astype("float64")).any():
        raise RuntimeError("e2s-query: a query row without its update row")
    es = sub["elapsed_seconds"].to_numpy().astype("float64")
    show = show_time_seconds(es, duration)
    cum = sub["elapsed_seconds_cumulative"].to_numpy().astype("float64") - es + show
    new = {"elapsed_seconds": show, "elapsed_seconds_cumulative": cum,
           "elapsed_seconds_sin": np.sin((show % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY),
           "elapsed_seconds_cos": np.cos((show % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY),
           "elapsed_seconds_cumulative_sin": np.sin((cum % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY),
           "elapsed_seconds_cumulative_cos": np.cos((cum % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY),
           "scaled_elapsed_seconds": np.asarray(dp.scale_elapsed_seconds(show)),
           "scaled_elapsed_seconds_cumulative": np.asarray(dp.scale_elapsed_seconds_cumulative(cum))}
    section_df = section_df.copy()
    for column in QUERY_INTERVAL_COLUMNS:
        section_df.loc[q, column] = new[column].astype(section_df[column].dtype)
    return section_df


@contextlib.contextmanager
def query_patch(variant):
    """e2s-query: rwkv.data_processing.add_queries rewrites the query rows' intervals while samples are built."""
    if variant != "e2s-query":
        yield
        return
    from rwkv import data_processing as dp
    original = dp.add_queries

    def add_queries(section_df, equalize_review_ths):
        return _query_rows_show_time(original(section_df, equalize_review_ths))

    dp.add_queries = add_queries
    try:
        yield
    finally:
        dp.add_queries = original


def samples(data, user, variant, deleted_cards="keep"):
    """The scorer's full-history samples for `variant` (compress_common.make_samples on the variant frame)."""
    from scripts import compress_common as cc
    from scripts import sm71_leak
    frame = load_frame(data, user, variant)
    if deleted_cards != "keep":
        frame, _ = sm71_leak.neutralise(frame, deleted_cards, data, user)
    frame = apply_frame(frame, variant)
    with query_patch(variant):
        return cc.make_samples(frame, user, len(frame))
