"""Render complete round summaries and compact handoffs from recorded results."""
import math
import os
from pathlib import Path
import re

from scripts.r01_metrics import case_metrics
from scripts.r01_workflow import read_json, write_json, write_text

LOSSES = {"curve_log_loss_label_weighted": "Curve label weighted", "curve_log_loss_user_average": "Curve user averaged",
          "instant_log_loss_label_weighted": "Instant label weighted", "instant_log_loss_user_average": "Instant user averaged"}


def exact_parity(parity):
    fields = parity.get("fields", {})
    return bool(fields) and all(fields.values()) and parity.get("equal_update_counts", False) and parity.get("update_mismatch_count", 1) == 0 and parity.get("exact_recorded_parity", True)


def entry(root, base, iteration, model):
    directory = Path(root) / base / f"iteration-{iteration}"
    comparison = read_json(directory / "comparison.json")
    item = comparison["comparisons"][model]
    case = (directory / model / item["track"]).resolve()
    if not case.is_relative_to(directory.resolve()):
        raise ValueError("Recorded case escapes its iteration directory")
    result = read_json(case / "result.json")
    if result.get("status") != "complete":
        raise ValueError(f"Iteration {iteration} is incomplete; do not close the round")
    protocol = read_json(directory / "protocol.json")
    previous = Path(root) / item["references"]["previous"]["path"]
    explanation = protocol.get("case_explanations", {}).get(model, {}).get(item["track"])
    plan = protocol.get("plan_path")
    plan_path = Path(root) / plan if plan else directory / "plan.json"
    if not explanation and plan_path.exists():
        plan_data = read_json(plan_path)
        import hashlib
        if not protocol.get("plan_sha256") or hashlib.sha256(plan_path.read_bytes()).hexdigest() == protocol["plan_sha256"]:
            explanation = plan_data.get("case_explanations", {}).get(model, {}).get(item["track"])
    explanation = explanation or f"Recorded candidate {item['track']}; see its frozen protocol and case report."
    isolation = read_json(directory / "source-isolation.json")
    return {"iteration": iteration, "track": item["track"], "exact": exact_parity(item["parity"]),
            "accepted": item["accepted_main"], "decision": item["decision"],
            "gain": item["references"]["previous"]["throughput_gain_percent"],
            "case": case, "previous": previous.parent, "origin": isolation["origin_git_commit"],
            "metrics": case_metrics(case), "comparison": item, "protocol": protocol, "result": result,
            "explanation": explanation, "background": read_json(directory / "background-load.json") if (directory / "background-load.json").exists() else None}


def relative(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def gain(value, reference):
    return 100 * (value / reference - 1)


def number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "— (not recorded)"


def count(value):
    return f"{value:,}" if isinstance(value, (int, float)) else "— (not recorded)"


def safe_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def round_documents(root, base, entries, commit, agent, summary, *, next_steps=()):
    """Render without mutating case results, raw logs, run-log schemas or decisions."""
    root, summary = Path(root), Path(summary)
    first, last = entries[0], entries[-1]
    reference = case_metrics(first["previous"])
    reference_iteration = first["previous"].parents[1].name.removeprefix("iteration-")
    accepted_path = root / last["comparison"]["accepted_reference"]["path"]
    accepted_result = read_json(accepted_path)
    accepted_iteration = accepted_path.parents[2].name.removeprefix("iteration-")
    accepted_metrics = case_metrics(accepted_path.parent)
    baseline = read_json(root / last["comparison"]["references"]["original_baseline"]["path"])
    authorizations = list(dict.fromkeys(item["protocol"].get("round_authorization", item["protocol"].get("authorization", "not recorded")) for item in entries))
    lines = [f"# {agent.capitalize()} R01 round: iterations {first['iteration']}–{last['iteration']}", "",
             f"The accepted preliminary {last['result'].get('model', accepted_path.parent.parent.name)} reference is **{accepted_iteration}: {accepted_metrics['reviews_per_second']:,.0f} reviews/s**, {gain(accepted_metrics['reviews_per_second'], reference['reviews_per_second']):+.2f}% versus the round’s starting accepted reference and {gain(accepted_metrics['reviews_per_second'], baseline['reviews_per_second']):+.2f}% versus the fixed original baseline. Implementation commit `{commit}` was made after the experiments.", "",
             "Authorization: " + " | ".join(authorizations) + ". The selected iteration budget is complete; another round requires a new human request.", "",
             "## Full-history results", "",
             "Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.", "",
             "| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]

    def row(iteration, track, metrics, change, decision, report):
        return (f"| [{iteration}]({relative(report, summary.parent)}) | {safe_cell(track)} | {number(metrics.get('training_seconds'))} | "
                f"{metrics['reviews_per_second']:,.0f} | {change} | {number(metrics.get('process_wall_seconds'))} | "
                f"{number(metrics.get('queue_wait_seconds'), 2)} | {number(metrics.get('preparation_seconds'), 1)} | {safe_cell(decision)} |")
    lines.append(row(reference_iteration, first["previous"].name, reference, "—", "Starting accepted reference", first["previous"].parents[1] / "summary.md"))
    for item in entries:
        lines.append(row(item["iteration"], item["track"], item["metrics"], f"{item['gain']:+.2f}%", item["decision"], item["case"].parents[1] / "summary.md"))
    lines.append("")
    for item in entries:
        r, q, m = item["result"], item["protocol"], item["metrics"]
        config, options = r.get("config", {}), r.get("options", {})
        labels = f"{r.get('training_curve_labels', 'not recorded')} Curve labels, {r.get('training_instant_labels', 'not recorded')} Instant labels"
        lines.extend([f"**{item['iteration']}, {item['track']}.** {item['explanation']} Recorded decision: {item['decision']}; exact full recorded parity: {item['exact']}. Training throughput changed {item['gain']:+.2f}% against its previous accepted reference.", "",
                      f"Exposure: {r.get('training_users', q.get('training_user_count', 'not recorded'))} training users, {r.get('evaluation_users', q.get('evaluation_users', 'not recorded'))} held-out users; {count(r.get('review_presentations'))} training reviews, {count(q.get('validation_reviews'))} evaluation reviews; {r.get('updates', 'not recorded')} updates; {labels}. Seed {config.get('SEED', 'not recorded')}, compute {config.get('DTYPE', 'not recorded')}, master precision {q.get('master_dtype', q.get('training', {}).get('master_dtype', 'not separately recorded'))}, epochs {config.get('EPOCHS', 'not recorded')}, batch chunks {options.get('batch_chunks', config.get('BATCH_CHUNKS', 'not recorded'))}, chunk reviews {config.get('CHUNK_REVIEWS', 'not recorded')}. Cohort hash `{q.get('cohort_sha256', 'not recorded')}`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.", ""])
        old = case_metrics(item["previous"])
        drift = [f"{label}: {m[key]:.2f} versus {old[key]:.2f}" for label, key in (("queue wait s", "queue_wait_seconds"), ("summed preparation s", "preparation_seconds"), ("validation s", "validation_seconds"), ("primer s", "discarded_primer_seconds"), ("worker CPU %", "worker_cpu_percent"), ("GPU activity %", "gpu_percent")) if m.get(key) is not None and old.get(key) is not None]
        lines.extend(["Resource observations, candidate versus its reference: " + "; ".join(drift) + ". These observations include machine-state variation and do not independently attribute a component’s effect.", ""])
    lines.extend(["## Separate held-out binary losses", "",
                  "| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |", "|---|---|---:|---:|---:|"])
    for item in entries:
        previous = read_json(item["previous"] / "result.json")
        original = read_json(root / item["comparison"]["references"]["original_baseline"]["path"])
        for key, label in LOSSES.items():
            value = item["result"]["evaluation"][key]
            lines.append(f"| {item['iteration']} | {label} | {value:.16g} | {gain(value, previous['evaluation'][key]):+.2f}% | {gain(value, original['evaluation'][key]):+.2f}% |")
    lines.extend(["", "## Checks and provenance", "",
                  f"Full recorded parity is exact in {sum(item['exact'] for item in entries)}/{len(entries)} cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.", ""])
    for item in entries:
        m, background = item["metrics"], item["background"]
        lines.append(f"- {item['iteration']}: experiment `{m['experiment_git_commit']}`, origin `{item['origin']}`, dirty at process start: {item['result'].get('provenance', {}).get('git_dirty', 'not recorded')}; [source archive]({relative(root / m['source_archive'], summary.parent)}), [protocol]({relative(item['case'].parents[1] / 'protocol.json', summary.parent)}), [comparison]({relative(item['case'].parents[1] / 'comparison.json', summary.parent)})." )
        if background:
            lines.append(f"- {item['iteration']} pre-run background: {background.get('total_cpu_percent', 'not recorded')}% total CPU; {background.get('python_gpu_processes', 'not recorded')} other Python GPU processes detected.")
    lines.extend(["", "## Pending next steps", ""])
    lines.extend(["- " + step for step in next_steps] if next_steps else ["- Select the next candidate from pending tracks in the optimization ledger and the latest host-path observations when a new round is authorized. No extra iteration or broader validation is authorized by this handoff."])
    lines.append("")
    next_iteration = max([int(item["iteration"]) for item in entries] + [int(path.name[10:]) for path in (root / base).glob("iteration-*") if re.fullmatch(r"iteration-\d+", path.name)]) + 1
    handoff = {"status": "completed preliminary round; new authorization required", "model": accepted_path.parent.parent.name,
               "iterations": [item["iteration"] for item in entries], "accepted_iteration": accepted_iteration,
               "accepted_result": accepted_path.relative_to(root).as_posix(), "accepted_options": accepted_result.get("options", {}),
               "accepted_reviews_per_second": accepted_metrics["reviews_per_second"], "next_iteration_hint": f"{next_iteration:04d}",
               "prior_miss_streak": last["comparison"].get("no_20_percent_gain_streak"),
               "implementation_git_commit": commit, "new_authorization_required": True,
               "host_path": "docs/R01_HOST_PATH.md", "ledger": "docs/GPU_OPTIMIZATIONS.md",
               "summary": summary.relative_to(root).as_posix(), "next_steps": list(next_steps),
               "acceptance": "Use current LOOP gates and source/runtime-compatible recorded checks; no broader validation authorized"}
    return "\n".join(lines), handoff


def write_documents(root, base, entries, commit, agent, summary, *, next_steps=(), refresh=False, update_index=True):
    summary = Path(summary)
    handoff_path = summary.with_name(summary.stem.removesuffix("-summary") + "-handoff.json")
    if summary.exists() and not refresh:
        print(f"[summary] kept existing {summary.relative_to(root)}")
    else:
        text, _ = round_documents(root, base, entries, commit, agent, summary, next_steps=next_steps)
        write_text(summary, text)
        print(f"[summary] complete {summary.relative_to(root)}")
    if not handoff_path.exists() or refresh:
        _, handoff = round_documents(root, base, entries, commit, agent, summary, next_steps=next_steps)
        write_json(handoff_path, handoff)
    print(f"[handoff] {handoff_path.relative_to(root)}")
    if update_index:
        readme = Path(root) / base / "README.md"
        lines = readme.read_text(encoding="utf-8").splitlines() if readme.exists() else ["# gpu-retrain-r01", "", "R01 screens and preserved raw evidence.", ""]
        index = next((i for i, line in enumerate(lines) if line.startswith("- **Open first:**")), None)
        previous = re.findall(r"\[([^]]+)\]\(([^)]+)\)", lines[index]) if index is not None else []
        new = f"- **Open first:** [Latest {agent} round, {entries[0]['iteration']}–{entries[-1]['iteration']}]({relative(summary, readme.parent)}); [compact handoff]({relative(handoff_path, readme.parent)})."
        prior = next(((label, target) for label, target in previous if target != relative(summary, readme.parent) and "handoff" not in label), None)
        if prior:
            new += f" Previous: [{prior[0]}]({prior[1]})."
        if index is None:
            lines.append(new)
        else:
            lines[index] = new
        write_text(readme, "\n".join(lines) + "\n")
    return handoff_path
