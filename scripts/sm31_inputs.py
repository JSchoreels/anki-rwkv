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

Again pressed as Hard (SM132, `again_as_hard`, scorer flag --again-as-hard P): on top of any variant, every review graded
Again (rating 1) is recorded as Hard (rating 2) in the MODEL INPUTS with probability P, independently per review: draw
u = numpy default_rng([132, user]).random(n_reviews) once per user, review review_th is flipped iff its rating is 1 and
u[review_th - 1] < P (deterministic per (user, P); the flipped sets are nested in P). Changed: the rating_1 / rating_2
one-hot of the review's update row and, with state "anki" (default), the card state Anki would have recorded (a Hard in
review state does not lapse, so the relearning rows that followed a flipped review-state Again are review-state rows;
state machine in _anki_states). Never changed: the `rating` / `y` columns and every label derived from them (the
targets stay true), the intervals (the real timeline), query rows (they carry no grade or state). The update row of
review t follows its own query row, so the corrupted grade of t only reaches predictions of later reviews.
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


AGAIN_AS_HARD_STATES = ("anki", "keep")


def again_as_hard_flips(frame, user, p):
    """SM132: boolean mask over the frame's rows: rating 1 and u[review_th - 1] < p, u = default_rng([132, user])."""
    review_th = frame["review_th"].to_numpy().astype(np.int64)
    u = np.random.default_rng([132, int(user)]).random(int(review_th.max()) if len(review_th) else 0)
    return (frame["rating"].to_numpy() == 1) & (u[review_th - 1] < p)


def _anki_states(state, rating, rating_c):
    """SM132: the card states of ONE card's rows (review order) under the corrupted grades rating_c (rating = true).

    Anki's states before each review: 0 new, 1 learning, 2 review, 3 relearning. Only an Again in review state lapses
    (-> relearning); a Hard there keeps the card in review. So a recorded relearning streak becomes review-state rows
    when the review-state Again that started it was flipped, until an (unflipped) Again in those rows lapses the card
    for real. A relearning streak with no recorded review-state Again before it keeps its states. Other states are
    kept. With no flipped row the result equals `state`."""
    out = []
    lapsed = False
    prev_s = prev_r = None
    for s, r, rc in zip(np.asarray(state).tolist(), np.asarray(rating).tolist(), np.asarray(rating_c).tolist()):
        if s == 3:
            if prev_s != 3 and not (prev_s == 2 and prev_r == 1):
                lapsed = True          # unexplained relearning start: keep the recorded states
            o = 3 if lapsed else 2
        else:
            o = s
            lapsed = False
        if o == 2 and rc == 1:
            lapsed = True              # an Again in review state lapses the card (also in the corrupted history)
        out.append(o)
        prev_s, prev_r = s, r
    return np.asarray(out, dtype=np.asarray(state).dtype)


def again_as_hard(frame, user, p, state="anki", flip=None):
    """SM132: (frame with the corrupted inputs, info, flipped review_ths, Again review_ths). The frame keeps its columns
    (add_queries lists them all). `flip` (checks only): an explicit row mask instead of the seeded draw."""
    if state not in AGAIN_AS_HARD_STATES:
        raise ValueError(f"again_as_hard state {state}")
    from rwkv import data_processing as dp
    if flip is None:
        flip = again_as_hard_flips(frame, user, p)
    else:
        flip = np.asarray(flip, dtype=bool) & (frame["rating"].to_numpy() == 1)
    rating = frame["rating"].to_numpy()
    review_th = frame["review_th"].to_numpy().astype(np.int64)
    info = {"p": p, "state": state, "seed": [132, int(user)], "reviews": int(len(frame)),
            "again": int((rating == 1).sum()), "flipped": int(flip.sum()),
            "again_review_state": int(((rating == 1) & (frame["state"].to_numpy() == 2)).sum()),
            "flipped_review_state": int((flip & (frame["state"].to_numpy() == 2)).sum()), "state_rows_changed": 0}
    frame = frame.copy()
    if flip.any():
        for k, v in ((1, 0), (2, 1)):
            col = f"rating_{k}"
            frame[col] = np.where(flip, v, frame[col].to_numpy()).astype(frame[col].dtype)
        if state == "anki":
            states = frame["state"].to_numpy()
            new = states.copy()
            rating_c = np.where(flip, 2, rating)
            cards = frame["card_id"].to_numpy()
            order = np.lexsort((review_th, cards))          # by card, then review order
            sc = cards[order]
            starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
            ends = np.r_[starts[1:], len(sc)]
            has_flip = np.add.reduceat(flip[order].astype(np.int64), starts) > 0 if len(sc) else []
            for a, b in zip(starts[has_flip], ends[has_flip]):
                idx = order[a:b]
                new[idx] = _anki_states(states[idx], rating[idx], rating_c[idx])
            changed = new != states
            info["state_rows_changed"] = int(changed.sum())
            if changed.any():
                frame["state"] = new.astype(frame["state"].dtype)
                frame["scaled_state"] = np.asarray(dp.scale_state(new.astype("float64"))).astype(frame["scaled_state"].dtype)
    return frame, info, review_th[flip], review_th[rating == 1]


def samples(data, user, variant, deleted_cards="keep", again_as_hard_p=None, again_as_hard_state="anki"):
    """The scorer's full-history samples for `variant` (compress_common.make_samples on the variant frame).
    SM132: with again_as_hard_p set, returns (samples, info, flipped review_ths, Again review_ths)."""
    from scripts import compress_common as cc
    from scripts import sm71_leak
    frame = load_frame(data, user, variant)
    if deleted_cards != "keep":
        frame, _ = sm71_leak.neutralise(frame, deleted_cards, data, user)
    frame = apply_frame(frame, variant)
    extra = None
    if again_as_hard_p is not None:
        frame, *extra = again_as_hard(frame, user, again_as_hard_p, again_as_hard_state)
    with query_patch(variant):
        out = cc.make_samples(frame, user, len(frame))
    return out if extra is None else (out, *extra)
