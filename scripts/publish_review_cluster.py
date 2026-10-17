"""Publish aggregate findings; exclude source event tuples and entity identifiers."""
from argparse import ArgumentParser
from hashlib import sha256
from html import escape
import json
from pathlib import Path


BUILDER = "https://github.com/open-spaced-repetition/anki-revlogs-dataset-builder/blob/e29946556796443283bf9c8e9d21f4502cb2e9c1/build_parquet.py"
EXPORT = "https://github.com/ankitects/anki/blob/main/docs-site/manual/exporting.mdx"
IMPORT = "https://github.com/ankitects/anki/blob/main/docs-site/manual/importing/packaged-decks.mdx"


def main():
    parser = ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--investigation", type=Path)
    source.add_argument("--from-summary", action="store_true",
                        help="only re-render cluster-95.html from the published cluster-95-summary.json")
    parser.add_argument("--output", type=Path, default=Path("docs/data-quality"))
    args = parser.parse_args()
    if args.from_summary:
        summary = json.loads((args.output / "cluster-95-summary.json").read_text(encoding="utf-8"))
        (args.output / "cluster-95.html").write_text(render(summary), encoding="utf-8")
        return
    raw_bytes = args.investigation.read_bytes()
    d = json.loads(raw_bytes)
    users = d["per_user"]
    reference_decks = {r["deck_id"]: r["parent_id"] for r in d["decks"][str(d["reference"]["user"])] if r["deck_id"] != 1}
    same_deck_ids = same_deck_tree = 0
    for rows in d["decks"].values():
        parents = {r["deck_id"]: r["parent_id"] for r in rows}
        same_deck_ids += reference_decks.keys() <= parents.keys()
        same_deck_tree += all(parents.get(k) == v for k, v in reference_decks.items())
    missing = [u for u in users if u["missing_reference"]]
    # Assert each aggregate interpretation against the private evidence.
    assert d["shared_distinct_events"] == d["reference"]["reviews"]
    assert all(u["missing_reference"] == u["missing_reviews_on_cards_with_later_learning_start"] for u in missing)
    assert all(u["partially_missing_reference_cards"] == 0 for u in missing)
    assert all(u["later_learning_reference_cards_without_missing_old_reviews"] == 0 for u in users)
    assert sum(u["extra_reference"] for u in users) == sum(u["private_reviews"] for u in users)
    provenance = d["provenance"]
    summary = {
        "status": "investigated; package-import mechanism inferred, original package unidentified",
        "scope": "Largest component of common-block links, minimum block 20, 95% coverage for either user, all groups.",
        "groups": len(d["groups"]), "users": len(d["users"]), "links": d["links"],
        "shared_distinct_events": d["shared_distinct_events"],
        "shared_presentations": d["shared_presentations"],
        "all_user_core_reviews": d["all_user_core"]["reviews"],
        "all_user_core_cards": d["all_user_core"]["cards"],
        "shared_events_minimum_owners": min(int(k) for k in d["ownership_histogram"] if int(k) >= 2),
        "private_reviews": sum(u["private_reviews"] for u in users),
        "users_with_full_reference": sum(u["missing_reference"] == 0 for u in users),
        "users_with_only_reference": sum(u["extra_reference"] == 0 for u in users),
        "users_with_later_extras_only": sum(u["extra_reference"] > 0 and u["extra_before_or_during_reference"] == 0 for u in users),
        "users_with_earlier_or_interleaved_extras": sum(u["extra_before_or_during_reference"] > 0 for u in users),
        "users_with_missing_old_reviews": len(missing),
        "missing_old_review_presentations": sum(u["missing_reference"] for u in missing),
        "missing_old_card_histories": sum(u["missing_reference_cards"] for u in missing),
        "all_missing_card_histories_have_later_learning_restart": True,
        "deck_hierarchy_size": len(reference_decks), "users_retaining_deck_ids": same_deck_ids,
        "users_retaining_deck_ids_and_parent_links": same_deck_tree,
        "deck_id_allocation_span_ms": max(reference_decks) - min(reference_decks),
        "deck_id_allocation_month": "2020-11", "shared_history_end_month": "2020-11",
        "users_retaining_2730_card_note_deck_mappings": sum(u["reference_cards_same_note"] >= 2730 and u["reference_cards_same_deck"] >= 2730 for u in users),
        "users_without_reference_card_metadata": sum(u["reference_card_count"] == 0 for u in users),
        "shared_event_years": d["shared_event_years"], "private_event_years": d["private_event_years"],
        "limits": ["No deck names, note text, package identity, import log or uploader identity in this export.",
                   "Entity IDs indicate allocation, not when each user imported a package.",
                   "A transferred collection, backup or copied profile can also preserve review history.",
                   "Common events establish shared history; they do not establish misconduct or that user identities are the same."],
        "verification": d["source_review_checks"],
        "provenance": {k: provenance[k] for k in ("captured_utc", "git_commit", "git_dirty", "source_tree_sha256")},
        "private_investigation_sha256": sha256(raw_bytes).hexdigest(),
        "sources": {"anki_export": EXPORT, "anki_import": IMPORT, "dataset_builder": BUILDER},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cluster-95-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "cluster-95.html").write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def render(summary):
    """The public page, from the aggregate summary alone (so it can be rebuilt from cluster-95-summary.json)."""
    rows = []
    maximum = max(summary["private_event_years"].values())
    for year in sorted(summary["shared_event_years"].keys() | summary["private_event_years"].keys()):
        shared = summary["shared_event_years"].get(year, 0)
        private = summary["private_event_years"].get(year, 0)
        rows.append(f'<tr><th>{year}</th><td>{shared:,}<span class="bar inherited" style="width:{shared / maximum * 100:.2f}%"></span></td><td>{private:,}<span class="bar personal" style="width:{private / maximum * 100:.2f}%"></span></td></tr>')
    s = summary
    src = s["sources"]
    provenance = s["provenance"]
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'">
<title>Why the large 95% review cluster shares history</title>
<meta name="description" content="Why 43 dataset users share 13,784 identical review events: an imported deck package that carried its author's review history.">
<link rel="stylesheet" href="../../style.css">
<style>
.eyebrow a{{color:inherit}}.lede a,main a,.foot a{{color:var(--accent)}}
:root{{--inherited:#bb682b;--personal:var(--accent)}}
.callout{{margin:12px 0 0;padding:14px 16px;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;background:var(--surface);max-width:none}}
.panel p,.panel li{{font-size:15px}}.panel p{{margin:8px 0 0}}.panel ul{{margin:8px 0 0;padding-left:20px}}.panel li{{margin-bottom:6px}}
.panel table{{width:100%;border-collapse:collapse;font-size:14px;margin-top:10px}}
.panel td,.panel th{{padding:8px;text-align:left;border-bottom:1px solid var(--line)}}
.panel thead th{{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.panel td{{width:44%;font-variant-numeric:tabular-nums}}
.bar{{display:block;height:7px;margin-top:4px;border-radius:2px}}.inherited{{background:var(--inherited)}}.personal{{background:var(--personal)}}
.key{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:baseline}}
.foot{{color:var(--muted);font-size:13px;padding:8px 16px 32px;overflow-wrap:anywhere}}code{{font-size:12px}}
@media(max-width:560px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<header class="top">
  <div class="wrap">
    <div class="top-row">
      <div>
        <p class="eyebrow"><a href="../../">anki-rwkv</a> &middot; <a href="../../factory.html">factory overview</a> &middot; <a href="../../models.html">model architectures</a> &middot; <a href="../../campaign.html">R01 campaign</a> &middot; <a href="../../train.html">8k/2k run</a> &middot; <a href="./">data quality</a></p>
        <h1>Why the large 95% cluster shares reviews</h1>
        <p class="lede"><a href="./">&larr; Review relation graph</a></p>
      </div>
      <button id="theme" class="btn" type="button" aria-label="Toggle light and dark theme">Theme</button>
    </div>
    <p class="callout"><strong>The evidence strongly fits an imported deck package containing its author's review history.</strong> Exact review events and an inherited deck hierarchy recur across many collections, followed by separate activity. A copied collection or backup remains an alternative; the export cannot identify the original package or author.</p>
  </div>
</header>
<main class="wrap">
<section class="cards" aria-label="Cluster summary">
<div class="card"><div class="k">users</div><div class="v">{s['users']}</div><div class="s">in {s['groups']} groups</div></div>
<div class="card"><div class="k">inherited reviews</div><div class="v">{s['shared_distinct_events']:,}</div><div class="s">distinct shared events</div></div>
<div class="card"><div class="k">in every user</div><div class="v">{s['all_user_core_reviews']:,}</div><div class="s">exact reviews present in all users</div></div>
<div class="card"><div class="k">individual reviews</div><div class="v">{s['private_reviews']:,}</div><div class="s">belonging to one user each</div></div>
</section>
<section class="panel">
<div class="panel-head"><h2>Which cluster this is</h2></div>
<p>This is the largest component under <strong>Common blocks · 95% · Either user · All groups</strong>. “Either” requires 95% overlap for at least one endpoint of each link. It does not mean that every pair of users shares 95% of both histories.</p>
</section>
<section class="panel">
<div class="panel-head"><h2>The evidence for an inherited package</h2></div>
<ul><li>The matches include the original answer timestamp, card ID, rating and duration. Sharing card content alone cannot explain these identical events.</li>
<li>Every one of the 13,784 shared events appears in 34–43 users. Together these events account for 588,219 review rows across the component.</li>
<li>{s['users_retaining_deck_ids']} users retain all {s['deck_hierarchy_size']} deck IDs from the reference hierarchy; {s['users_retaining_deck_ids_and_parent_links']} retain its parent links too. The IDs were allocated within {s['deck_id_allocation_span_ms']} milliseconds in November 2020, shortly after the inherited review history ends. This is an allocation clue, not an individual import date.</li>
<li>{s['users_retaining_2730_card_note_deck_mappings']} users retain the same 2,730 reviewed card IDs with the same note and deck assignments.</li>
<li>Four histories contain only the inherited events. Of the other 39 users, 35 add reviews only after the inherited history ends; four also have earlier or interleaved activity. All 200,511 events outside the inherited set are unique to one user within this component.</li></ul>
<p>Anki explicitly supports exporting review history with scheduling information. Its import documentation describes adopting another deck author's learning progress. That provides a supported mechanism for this pattern. <a href="{src['anki_export']}">Export documentation</a> · <a href="{src['anki_import']}">Import documentation</a></p>
</section>
<section class="panel">
<div class="panel-head"><h2>Why some histories match in blocks instead of containing the whole history</h2></div>
<p>Twenty-one users are missing some inherited reviews. Across them, all {s['missing_old_review_presentations']:,} missing rows belong to {s['missing_old_card_histories']:,} whole card histories. Every affected card has a later learning restart; there are no partially missing old card histories and no restart exceptions.</p>
<p>The <a href="{src['dataset_builder']}">exact dataset builder revision</a> keeps only the last learning sequence of each card. Its filtering explains the observed gaps: restarting a card can remove its inherited history from this export, splitting long chronological matches into separate blocks. The filtered export alone cannot prove whether every missing row remains in the original raw collection.</p>
</section>
<section class="panel">
<div class="panel-head"><h2>Shared history versus individual activity</h2><p class="hint">Each distinct event is counted once. <i class="key inherited"></i>shared by multiple users, <i class="key personal"></i>present in one user only. UTC calendar-year aggregates.</p></div>
<div class="table-wrap"><table><thead><tr><th>Year</th><th>Shared events</th><th>Individual events</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</section>
<section class="panel">
<div class="panel-head"><h2>What we can use next</h2></div>
<p>The 13,784-event signature, its ownership across users, the common deck hierarchy and later learning restarts identify an inherited history family. For data quality, flag shared review events and preserve the later personal reviews. Excluding entire users would discard 200,511 unique reviews; no filtering or training experiment has been performed.</p>
<p>To name the deck or confirm its distribution route, we would need deck names, note content, an original package or import records. This export contains only entity IDs and their relationships. In 14 users, none of the reference reviewed cards remains in current card metadata; review logs can survive deletion, so metadata absence does not refute the shared history.</p>
</section>
</main>
<footer class="wrap foot">All 43 review source checksums were checked against the full audit, and canonical events rebuilt from Parquet matched the cached arrays exactly. Public output contains aggregate findings only. <a href="cluster-95-summary.json">Machine-readable findings</a>.<br>Run source commit: <code>{escape(provenance['git_commit'])}</code>; dirty source snapshot: <code>{escape(provenance['source_tree_sha256'])}</code>.</footer>
<script src="../../common.js"></script>
<script src="../../nav.js" defer></script>
</body>
</html>'''


if __name__ == "__main__":
    main()
