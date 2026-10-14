"""Opt-in interval definition for the published revlogs (docs/EXPERTIUM_CATCH_UP.md item 4).

rwkv.data_processing.get_rwkv_data applies `end_to_start_published` when EXPERTIUM_END_TO_START=1; the R01 track
option `"end_to_start_intervals": true` sets that variable before any data is read or worker is spawned.
Ported from Expertium/rwkv-anki-autoresearch@a9fcf466 rwkv/id_features.elapsed_end_to_start_published.
"""


def end_to_start_published(df):
    """Item 4: the interval to a review measured from the previous answer to this review's START.

    The published revlogs' `elapsed_seconds` runs answer-to-answer, so it contains the current review's own
    `duration`, which is not known before the review and correlates with the outcome (upstream
    rwkv/id_features.elapsed_end_to_start_published, RWKV_E2S_PUBLISHED). `duration` is in milliseconds; the -1
    first-review sentinel is kept and negatives are clamped to 0 before the integer cast. Every row moves, so the
    real rows, the query rows and the curve label times stay consistent; elapsed_days is a calendar-day index and
    is left alone, as upstream does.
    """
    import numpy as np
    if "elapsed_seconds" not in df.columns or "duration" not in df.columns or not len(df):
        return df
    seconds = df["elapsed_seconds"].to_numpy().astype("float64")
    duration = df["duration"].to_numpy().astype("float64") / 1000.0
    corrected = np.floor(np.maximum(seconds - duration, 0.0))
    corrected[seconds == -1] = -1.0
    df = df.copy()
    df["elapsed_seconds"] = corrected.astype(df["elapsed_seconds"].dtype)
    return df
