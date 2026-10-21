"""SM59 final table: arms kd-control/kd-090/kd-100 (and SM50's curve-only / instant-only arms if their summaries exist) vs SM36.
Reads summary-<arm>.json written by the runbooks. Writes summary-table.md next to this file."""
import json
from pathlib import Path

here = Path(__file__).parent
sm50 = Path("C:/Users/jonat/workspace/wt/sm50/artifacts/small-model/sm50")
arms = [("SM59 kd-control (teacher weight 0.5)", here / "summary-kd-control.json"),
        ("SM59 kd-090 (0.9)", here / "summary-kd-090.json"),
        ("SM59 kd-100 (1.0)", here / "summary-kd-100.json"),
        ("SM50 curve head only", sm50 / "summary-curve.json"),
        ("SM50 instant head only", sm50 / "summary-instant.json")]
lines = ["# SM59 / SM50 arms vs SM36 (w8+s8, review-weighted LogLoss, paired user bootstrap 95% CI of arm - SM36)", ""]
for pool in ("732", "464"):
    lines += [f"## {pool} pool", "", "| arm | users | curve LogLoss | curve delta vs SM36 [95% CI] | curve % vs fork | instant LogLoss | instant delta vs SM36 [95% CI] | instant % vs fork |",
              "|---|---|---|---|---|---|---|---|"]
    base = None
    for name, path in arms:
        if not path.exists():
            lines.append(f"| {name} | not available | | | | | | |")
            continue
        r = json.loads(path.read_text())[pool]
        c, i = r["curve"], r["instant"]
        base = base or (r["users"], c["sm36"], c["sm36_pct_vs_fork"], i["sm36"], i["sm36_pct_vs_fork"])
        lines.append(f"| {name} | {r['users']} | {c['arm']:.5f} | {c['delta']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] | {c['arm_pct_vs_fork']:+.2f}% | "
                     f"{i['arm']:.5f} | {i['delta']:+.5f} [{i['ci95'][0]:+.5f}, {i['ci95'][1]:+.5f}] | {i['arm_pct_vs_fork']:+.2f}% |")
    if base:
        lines.append(f"| SM36 (reference) | {base[0]} | {base[1]:.5f} | | {base[2]:+.2f}% | {base[3]:.5f} | | {base[4]:+.2f}% |")
    lines.append("")
(here / "summary-table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
