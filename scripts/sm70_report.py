"""SM70: ranked Markdown tables from the probe JSONs (within-user offset probe = primary, pooled probe and the big model
as supplementary). usage: sm70_report.py --dir artifacts/small-model/sm70 > tables.md"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm70_features import FEATURES  # noqa: E402

# plain description, deployable (yes / no + why), implementation cost
INFO = {
    "s_state": ("review type of this review (learning / review / relearning / filtered)", "yes", "low: unmask the state column on query rows"),
    "c_n": ("number of earlier reviews of the card", "yes", "none: the card stream counts them"),
    "c_lapses": ("lapses so far (Again in review state)", "yes", "low: one input column"),
    "c_fail_frac": ("share of Again among the card's labelled reviews", "yes", "low"),
    "c_streak": ("successes since the card's last Again", "yes", "low"),
    "c_hard_frac": ("share of Hard among the card's successes", "yes", "low"),
    "c_easy_frac": ("share of Easy among the card's successes", "yes", "low"),
    "c_entropy": ("entropy of the card's rating history", "yes", "low"),
    "c_growth": ("this interval / the previous interval (log)", "yes", "low"),
    "c_max_ratio": ("this interval / the longest earlier interval (log)", "yes", "low"),
    "c_age": ("days since the card's first review", "yes", "none: elapsed_days_cumulative is an input"),
    "c_today": ("earlier reviews of the card today", "yes", "low"),
    "c_dur_mean": ("mean answer time of the card's earlier reviews", "yes", "low"),
    "c_dur_last": ("answer time of the card's previous review", "yes", "none: duration is an input"),
    "c_first_rating": ("rating of the card's first review", "yes", "low"),
    "n_size": ("cards in the note (export-time cards table)", "snapshot: export-time table", "low (Anki knows the note's cards)"),
    "n_sib_n": ("earlier reviews of sibling cards", "yes", "low"),
    "n_sib_fail": ("Again share of sibling reviews", "yes", "low"),
    "n_sib_gap": ("days since a sibling was reviewed", "yes", "low"),
    "n_sib_last_y": ("result of the last sibling review", "yes", "low"),
    "d_size": ("cards in the deck (export-time table)", "snapshot: export-time table", "low"),
    "d_depth": ("depth of the deck in the hierarchy", "snapshot: export-time table", "low"),
    "p_ndecks": ("decks sharing the preset", "snapshot: export-time table", "low"),
    "p_size": ("cards under the preset (export-time table)", "snapshot: export-time table", "low"),
    "d_ret": ("deck's running recall rate (other cards)", "yes", "low"),
    "d_today": ("reviews in the deck today so far", "yes", "low"),
    "k_rank": ("position of the card in creation order (export-time)", "snapshot: uses later cards", "low"),
    "k_order_gap": ("share of earlier-learned cards created after this one", "yes", "medium: needs creation order"),
    "k_intro_jump": ("creation-order jump from the previously introduced card", "yes", "medium"),
    "k_nb_ret": ("recall rate of cards created next to this one", "yes", "medium: needs creation order"),
    "k_nb_n": ("labelled reviews of cards created next to this one", "yes", "medium"),
    "k_backlog": ("cards never studied yet (export-time table)", "snapshot: export-time table", "low"),
    "u_n": ("user's reviews so far", "yes", "none: the user stream counts them"),
    "u_ret": ("user's running recall rate", "yes", "low"),
    "u_ret_rev": ("user's running recall rate in review state", "yes", "low"),
    "u_ret_100": ("user's recall rate over the last 100 labelled reviews", "yes", "low"),
    "u_today_fail": ("user's Again share today so far", "yes", "low"),
    "u_gap": ("days since the user's previous study day", "yes", "low"),
    "u_yday": ("reviews on the previous study day", "yes", "low"),
    "u_load": ("today's reviews so far / the user's mean daily reviews", "yes", "low"),
    "u_hard": ("user's share of Hard among successes", "yes", "low"),
    "u_easy": ("user's share of Easy among successes", "yes", "low"),
    "u_wd_ret": ("user's recall rate on this weekday minus overall", "yes", "low"),
    "u_dur_last": ("answer time of the user's previous review (any card)", "yes", "low"),
    "u_dur_mean": ("user's mean answer time", "yes", "low"),
    "t_tod": ("local time of day (from review times)", "yes (Anki has the clock)", "medium: review timestamps in training data (-id build) and at inference"),
    "t_sib_gap_h": ("hours since a sibling was shown", "yes", "medium: -id build"),
    "t_gap_prev": ("pause before the previous review (past only)", "yes", "medium: -id build"),
    "t_break": ("break before this review, if over 30 min", "yes", "medium: -id build"),
    "t_sess_pos_prev": ("position in the session (as of the previous review)", "yes", "medium: -id build"),
    "t_sess_min_prev": ("minutes into the session (as of the previous review)", "yes", "medium: -id build"),
    "t_sess_fail_prev": ("Again share in the session (as of the previous review)", "yes", "medium: -id build"),
    "t_since_any": ("pause between the previous answer and this card's estimated show time", "no: the show time is answer time minus the recorded (capped / paused) duration of this review", "-"),
    "t_sess_pos": ("position in the session (boundary from this review's show time)", "no: same show-time leak", "-"),
    "t_sess_min": ("minutes into the session (this review's show time)", "no: same show-time leak", "-"),
    "t_sess_fail": ("Again share in the session (boundary from this review's show time)", "no: same show-time leak", "-"),
    "r_card_age": ("days since the card was created (raw Anki id)", "yes", "medium: -id build"),
    "r_note_age": ("days since the note was created", "yes", "medium: -id build"),
    "r_deck_age": ("days since the deck was created", "yes", "medium: -id build"),
    "r_wait_new": ("days between creation and first review", "yes", "medium: -id build"),
    "r_batch": ("cards created within 60 s of this one (import / batch size)", "yes", "medium: -id build"),
}


def fmt(c):
    return f"{c['delta']:+.6f} [{c['ci'][0]:+.6f}, {c['ci'][1]:+.6f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    w = json.loads((d / "within-sm56.json").read_text())
    wf = json.loads((d / "within-fork.json").read_text()) if (d / "within-fork.json").exists() else None
    pooled = json.loads((d / "pooled-sm56.json").read_text()) if (d / "pooled-sm56.json").exists() else None
    pf = json.loads((d / "pooled-fork.json").read_text()) if (d / "pooled-fork.json").exists() else None
    names = [n for n in w["heads"]["instant"]["offset"]]
    wc, wi = w["heads"]["curve"]["offset"], w["heads"]["instant"]["offset"]
    rank = sorted(names, key=lambda n: min(wc[n]["delta"], wi[n]["delta"]))
    print(f"Within-user probe (primary), SM56, {w['users']} users, {w['eval_reviews']:,} evaluation reviews "
          f"(second halves). Delta = LogLoss with the feature - without, review-weighted; 95% user bootstrap.\n")
    print("| Rank | Feature | Plain description | Curve delta | Instant delta | Interval (curve; instant) | Deployable | Implementation cost |")
    print("|---:|---|---|---:|---:|---|---|---|")
    for i, n in enumerate(rank, 1):
        if n.startswith("control:"):
            desc, dep, cost = "positive control: the big model's own prediction", "-", "-"
            label = "big-model prediction (control)"
        elif n.startswith("group:"):
            g = n.split(":", 1)[1]
            desc, dep, cost = f"all {g} features together", "see rows", "see rows"
            label = f"{g} group"
        else:
            desc, dep, cost = INFO.get(n, ("", "", ""))
            label = n
        ci = f"[{wc[n]['ci'][0]:+.6f}, {wc[n]['ci'][1]:+.6f}]; [{wi[n]['ci'][0]:+.6f}, {wi[n]['ci'][1]:+.6f}]"
        print(f"| {i} | {label} | {desc} | {wc[n]['delta']:+.6f} | {wi[n]['delta']:+.6f} | {ci} | {dep} | {cost} |")
    print()
    if pooled or wf:
        print("Supplementary: pooled cross-user probe (SM56) and the same probes on the big model.\n")
        print("| Feature | Pooled SM56 curve | Pooled SM56 instant | Within big model curve | Within big model instant | Pooled big model curve | Pooled big model instant |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for n in rank:
            row = []
            for src, kind, head in ((pooled, "cands", "curve"), (pooled, "cands", "instant"), (wf, "offset", "curve"),
                                    (wf, "offset", "instant"), (pf, "cands", "curve"), (pf, "cands", "instant")):
                c = src["heads"].get(head, {}).get(kind, {}).get(n) if src else None
                row.append(fmt(c) if c else "-")
            print(f"| {n} | " + " | ".join(row) + " |")
        if pooled:
            print()
            for head in ("curve", "instant"):
                H = pooled["heads"].get(head)
                if not H:
                    continue
                extra = {k: v for k, v in H["cands"].items() if k.startswith(("tree:", "all-"))}
                print(f"Pooled {head}: raw {H['raw_logloss']:.5f}, Platt {H['platt_logloss']:.5f}; " +
                      "; ".join(f"{k} {fmt(v)}" for k, v in extra.items()))
    for head in ("curve", "instant"):
        H = w["heads"][head]
        print(f"\nWithin {head}: raw {H['raw_logloss']:.5f}, per-user Platt {H['user_platt_logloss']:.5f} "
              f"(Platt - raw {H['user_platt_minus_raw']:+.6f} [{H['user_platt_minus_raw_ci'][0]:+.6f}, "
              f"{H['user_platt_minus_raw_ci'][1]:+.6f}])")


if __name__ == "__main__":
    main()
