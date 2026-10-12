"""Read local session records; emit aggregate usage without message bodies."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "token-audit.json"
THREAD = os.environ["CODEX_THREAD_ID"]
records = {}
for path in Path("C:/Users/jonat/.codex/sessions/2026/10/12").glob("rollout-*.jsonl"):
    with path.open(encoding="utf-8") as stream:
        try:
            first = json.loads(next(stream))
        except (StopIteration, json.JSONDecodeError):
            continue
        if first.get("type") != "session_meta":
            continue
        meta = first["payload"]
        records[meta["id"]] = {"path": path, "parent": meta.get("parent_thread_id"),
                               "source": meta.get("source"), "agent_path": meta.get("agent_path")}
selected = {THREAD}
while True:
    additions = {key for key, value in records.items() if value["parent"] in selected}
    if additions <= selected:
        break
    selected |= additions
threads = []
for key in sorted(selected):
    record = records[key]
    calls = {}
    usage = None
    completed = 0
    for line in record["path"].open(encoding="utf-8"):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = item.get("payload", {})
        if item.get("type") == "token_usage_record" and data.get("thread_id") == key:
            usage = data.get("thread_token_usage")
        if item.get("type") == "response_item" and data.get("type") in ("function_call", "custom_tool_call"):
            calls[data.get("call_id") or data.get("id")] = data
        if item.get("type") == "item_completed" and data.get("item", {}).get("type") == "command_execution":
            completed += 1
    kinds = Counter(value.get("name", "unknown") for value in calls.values())
    nested = Counter()
    for value in calls.values():
        code = value.get("input", value.get("arguments", ""))
        if not isinstance(code, str):
            code = json.dumps(code)
        if value.get("name", "").split(".")[-1] == "exec":
            nested.update(re.findall(r"tools\.(\w+)\s*\(", code))
    normalized = dict(usage or {})
    if usage:
        normalized["uncached_input_tokens"] = usage["input_tokens"] - usage.get("cached_input_tokens", 0)
    source = record["source"]
    role = "root" if key == THREAD else "approval_reviewer" if any(label in json.dumps(source).lower() for label in ("review", "guardian")) else "work_agent"
    threads.append({"thread_id": key, "parent_thread_id": record["parent"], "role": role,
                    "agent_path": record["agent_path"], "usage": normalized,
                    "logged_tool_calls": len(calls), "tool_names": dict(kinds), "nested_tool_calls": dict(nested),
                    "shell_operations": nested["exec_command"], "process_polls": nested["write_stdin"],
                    "tool_wait_resumptions": sum(n for name, n in kinds.items() if name.split(".")[-1] == "wait"),
                    "completed_command_execution_events": completed})
def aggregate(items):
    usage = Counter()
    for item in items:
        usage.update(item["usage"])
    return {"usage": dict(usage), **{field: sum(item[field] for item in items)
             for field in ("logged_tool_calls", "shell_operations", "process_polls", "tool_wait_resumptions")}}
result = {"root_thread_id": THREAD, "recorded_utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Latest cumulative local thread usage at audit cutoff; root, delegated work and approval reviewers itemized separately",
          "status": "cutoff snapshot; subsequent closeout and final-answer usage excluded",
          "threads": threads, "aggregate": aggregate(threads),
          "root": aggregate([item for item in threads if item["role"] == "root"]),
          "work_agents_aggregate": aggregate([item for item in threads if item["role"] == "work_agent"]),
          "approval_reviewers_aggregate": aggregate([item for item in threads if item["role"] == "approval_reviewer"]),
          "baseline": "artifacts/diagnostics/r01-round-0034-0036-options/token-audit.json",
          "limitations": ["Workload differs: this round introduces two CPU implementations and a new CUDA kernel; prior round retuned options.",
                           "Observed counts do not establish causality from helpers or model choice.",
                           "Nested tool counts are regex reconstructions, not an independently verified event taxonomy.",
                           "Tool-output characters unavailable; post-cutoff usage excluded.",
                           "Only attributable descendant approval reviewers included; absence means none found in local records, not globally zero usage."]}
prior = json.loads((ROOT / result["baseline"]).read_text(encoding="utf-8"))
result["comparison"] = {"prior_root": next(value for value in prior["threads"].values() if value["role"] == "root"),
                        "current_root": result["root"], "causal_savings": "inconclusive; unlike workloads and audit cutoffs"}
OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: result[key] for key in ("root_thread_id", "recorded_utc", "root", "work_agents_aggregate", "approval_reviewers_aggregate")}))
