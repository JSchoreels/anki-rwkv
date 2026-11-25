#!/usr/bin/env python
"""Weekend RWKV family release (SM166): build the six package folders, their zips, INTEGRATION.md, WEIGHTS-NOTICE.md
and SHA256SUMS into a local release folder. Preparation only: it calls no gh command, creates no tag or release and
uploads nothing.

    python -X utf8 artifacts/small-model/sm166/build_release.py [--out work/release-2026-11-28]

Before: time_vectors.py has written <out>/time-vectors/<key>/time_user_*.npz for every key (see run_all.sh), and
<out>/fork-main/ holds the fork main files (run_all.sh extracts them with git cat-file from fork main eda153ddb).
After: verify_release.py extracts every zip into an empty folder and runs its checks.

Layout (as the 13 November release rwkv-family-2026-11-13): one zip per model with one top-level folder named after the
zip; INTEGRATION.md (the Small two-head package's), WEIGHTS-NOTICE.md and SHA256SUMS beside the zips.
"""
from argparse import ArgumentParser
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WT = Path("C:/Users/jonat/workspace/wt")
FORK = Path("C:/Users/jonat/workspace/anki")
FORK_COMMIT = "eda153ddbff7847de8bf2591fe5e0935a8dfe836"
FORK_INFER = "qt/aqt/rwkv_inference"
HF = ROOT / "artifacts/hf-family/repo-new"          # the 13 November release's manuals (read only)
TAG = "rwkv-family-2026-11-28"                       # proposed tag, not created
ZIP_TIME = (2026, 11, 28, 0, 0, 0)

# Reference LogLoss of the 13 November table: the big model (1 leak-free epoch), show-time inputs, w8+s8, 722 users.
BIG_REF = {"curve": 0.29958, "instant": 0.277157}

PACKAGES = [
    {"zip": "rwkv-big-two-head", "key": "sm91s2", "size": "Big", "variant": "two-head",
     "name": "the show-time big model (SM91s2)", "label": "sm91s2-big-showtime",
     "src": WT / "big-package/artifacts/models/sm91s2", "base": HF / "rwkv-big-two-head/INTEGRATION.md",
     "pred": "the same model (the 13 November package rwkv-big-two-head)",
     "fork_weights": "RWKV_trained_on_5000_10000.bin", "fork_master": "RWKV_trained_on_5000_10000.pth",
     "fork_int8": None, "fork_head": "RWKV_trained_on_5000_10000.time_head",
     "logloss": {"curve": 0.299415, "instant": 0.276183},
     "fmt": "F6", "sm164": {"curve": "+0.043% [+0.032, +0.056]", "instant": "+0.069% [+0.063, +0.076]"},
     "budget": {"curve": "+0.055% (budget +0.15% vs its own int8)", "instant": "+0.077% (budget +0.15%)"},
     "heads": "curve, instant and rating heads (a full model: forgetting curve, intervals, 90% stability, instant recall"
              " and buttons) plus the answer-time head",
     "what": "The Anki fork's big architecture (2,762,884 parameters) trained on show-time inputs, leak-free. Same model,"
             " same weight files and same test vectors as in the 13 November release; new in this package: the "
             "answer-time head and the fork's new state format (sections A.3 and A.4)."},
    {"zip": "rwkv-big-instant-only", "key": "sm99", "size": "Big", "variant": "Instant-Only",
     "name": "the show-time instant-only big model (SM99)", "label": "sm99-big-inst-e2",
     "src": WT / "big-package/artifacts/models/sm99", "base": HF / "rwkv-big-instant-only/INTEGRATION.md",
     "pred": "the same model (the 13 November package rwkv-big-instant-only)",
     "fork_weights": "sm99/weights.f32.bin", "fork_master": None, "fork_int8": None, "fork_head": "sm99/time_head",
     "logloss": {"instant": 0.275894},
     "fmt": "F6", "sm164": {"instant": "+0.074% [+0.067, +0.081]"},
     "budget": {"instant": "+0.084% (budget +0.15% vs its own int8)"},
     "heads": "instant and rating heads only (the curve head is untrained: never use p_curve, intervals or the 90%"
              " stability) plus the answer-time head",
     "what": "The show-time big model (SM91s2) trained further with the curve loss off. Same model, same weight files and"
             " same test vectors as in the 13 November release; new in this package: the answer-time head and the"
             " fork's new state format (sections A.3 and A.4)."},
    {"zip": "rwkv-small-two-head", "key": "sm118", "size": "Small", "variant": "two-head",
     "name": "the show-time two-head cool-down for Small (SM118)", "label": "sm118-e26-dec2ep32k-t91s2",
     "src": WT / "fork-rc-export/artifacts/models/sm118", "base": HF / "rwkv-small-two-head/INTEGRATION.md",
     "pred": "the show-time retrain (SM92), package sm92 of the 13 November release",
     "fork_weights": "sm118/weights.f32.bin", "fork_master": None, "fork_int8": "sm118/weights.int8.bin",
     "fork_head": "sm118/time_head",
     "logloss": {"curve": 0.300016, "instant": 0.276599},
     "fmt": "F6", "sm164": {"curve": "+0.034% [+0.027, +0.041]", "instant": "+0.061% [+0.056, +0.067]"},
     "budget": {"curve": "+0.412% (budget +0.5% vs the show-time big model (SM91s2) at int8)",
                "instant": "+0.243% (budget +0.5%)"},
     "heads": "curve, instant and rating heads (a full model) plus the answer-time head",
     "what": "The show-time retrain (SM92)'s constant-learning-rate epoch 26 with a two-epoch 32k Muon cool-down in two"
             " passes, distilled from the show-time big model (SM91s2) (both heads, show-time inputs, leak-free)."
             " Architecture w64 (331,511 parameters), the same as the show-time retrain (SM92): the weight file layout,"
             " tensor names, state and every section of Part B apply unchanged; only the values differ. Against the"
             " show-time retrain (SM92): curve -0.113% [-0.176, -0.062], instant -0.100% [-0.110, -0.091] (722 users,"
             " w8+s8)."},
    {"zip": "rwkv-small-instant-only", "key": "sm113", "size": "Small", "variant": "Instant-Only",
     "name": "the longer Small Instant-Only model (SM113)", "label": "sm113-inst-e35st-st-dec2ep32k-t99",
     "src": WT / "time-heads-package/artifacts/models/sm113", "base": HF / "rwkv-small-instant-only/INTEGRATION.md",
     "pred": "the Small Instant-Only model (SM98), package sm98 of the 13 November release",
     "fork_weights": "sm113/weights.f32.bin", "fork_master": None, "fork_int8": "sm113/weights.int8.bin",
     "fork_head": "sm113/time_head",
     "logloss": {"instant": 0.275933},
     "fmt": "F6", "sm164": {"instant": "+0.067% [+0.060, +0.073]"},
     "budget": {"instant": "+0.116% (budget +0.5% vs the show-time instant-only big model (SM99) at int8)"},
     "heads": "instant and rating heads only (the curve head is untrained: never use p_curve, intervals or the 90%"
              " stability) plus the answer-time head",
     "what": "The Small Instant-Only model (SM98)'s line with a two-epoch 32k Muon cool-down in two passes, distilled"
             " from the show-time instant-only big model (SM99) (curve loss off, show-time inputs, leak-free)."
             " Architecture w64 (331,511 parameters), the same as SM98: every section of Part B applies unchanged;"
             " only the values differ. Against the Small Instant-Only model (SM98): instant -0.094% [-0.103, -0.085]"
             " (722 users, w8+s8). Its manifest's `use` text still says it runs beside the show-time retrain (SM92);"
             " in the fork it now runs beside the show-time two-head cool-down for Small (SM118)."},
    {"zip": "rwkv-tiny-two-head", "key": "sm130th", "size": "Tiny", "variant": "two-head",
     "name": "the six-layer Tiny two-head model (SM130)", "label": "sm130-th-dec2ep32k-t91s2",
     "src": WT / "fork-rc-export/artifacts/models/sm130th", "base": HF / "rwkv-tiny-two-head/INTEGRATION.md",
     "pred": "the Tiny two-head model (SM106), package sm106 of the 13 November release",
     "fork_weights": "sm130th/weights.f32.bin", "fork_master": None, "fork_int8": "sm130th/weights.int8.bin",
     "fork_head": "sm130th/time_head",
     "logloss": {"curve": 0.301416, "instant": 0.278049},
     "fmt": "T24", "sm164": {"curve": "+0.040% [+0.031, +0.049]", "instant": "+0.074% [+0.067, +0.082]"},
     "budget": {"curve": "+0.908% (budget +1.0% vs the show-time big model (SM91s2) at int8)",
                "instant": "+0.806% (budget +1.0%)"},
     "heads": "curve, instant and rating heads (a full model) plus the answer-time head",
     "what": "The Tiny two-head chain (the Tiny two-head model (SM106)'s constant-learning-rate epoch 29) cut to six"
             " layer steps, three recovery epochs with both heads, then a two-pass 32k Muon cool-down distilled from"
             " the show-time big model (SM91s2) (show-time inputs, leak-free). Against the show-time two-head"
             " cool-down for Small (SM118): curve +0.467% [+0.398, +0.546], instant +0.524% [+0.484, +0.567] (722"
             " users, w8+s8), inside the Tiny budget of +1% vs Small.",
     "six_layer": True},
    {"zip": "rwkv-tiny-instant-only", "key": "sm130io", "size": "Tiny", "variant": "Instant-Only",
     "name": "the six-layer Tiny Instant-Only model (SM130)", "label": "sm130-io-dec2ep32k-t99",
     "src": WT / "fork-rc-export/artifacts/models/sm130io", "base": HF / "rwkv-tiny-instant-only/INTEGRATION.md",
     "pred": "the Tiny Instant-Only model (SM101), package sm101 of the 13 November release",
     "fork_weights": "sm130io/weights.f32.bin", "fork_master": None, "fork_int8": "sm130io/weights.int8.bin",
     "fork_head": "sm130io/time_head",
     "logloss": {"instant": 0.27749},
     "fmt": "T24", "sm164": {"instant": "+0.089% [+0.081, +0.099]"},
     "budget": {"instant": "+0.720% (budget +1.0% vs the show-time instant-only big model (SM99) at int8)"},
     "heads": "instant and rating heads only (the curve head is untrained: never use p_curve, intervals or the 90%"
              " stability) plus the answer-time head",
     "what": "The Tiny Instant-Only chain (the Tiny Instant-Only model (SM101)'s constant-learning-rate epoch 38) cut to"
             " six layer steps, three recovery epochs with the curve loss off, then a two-pass 32k Muon cool-down"
             " distilled from the show-time instant-only big model (SM99) (show-time inputs, leak-free). Against the"
             " longer Small Instant-Only model (SM113): instant +0.564% [+0.522, +0.607] (722 users, w8+s8), inside"
             " the Tiny budget of +1% vs Small.",
     "six_layer": True},
]

# files copied from a source package folder (everything else, such as logs, stays out)
COPY = ["architecture.json", "manifest.json", "parity.json", "weights.f32.bin", "weights.int8.bin", "master.pth",
        "reference/run_reference.py", "reference/small_model_numpy.py", "reference/small_model_recurrent.py",
        "reference/weights_io.py", "testvectors/check_testvectors.py", "testvectors/make_testvectors.py",
        "testvectors/user_900001.npz", "testvectors/user_900002.npz"]

FORMATS = {
    "F6": {"label": "w8+lr2.16.4q6x8", "byte": "0x26", "factors": "int6 factor codes (a float16 scale of max/31 per"
           " factor vector, codes in [-31, 31], bit-packed four codes in three bytes)"},
    "T24": {"label": "w8+lr2.16.4q8x8", "byte": "0x28", "factors": "int8 factor codes (a float16 scale of max/127 per"
            " factor vector, codes in [-127, 127])"},
}
# card + note state bytes per size (fork layout; one model): int8 rows and T24 means from the fork's
# docs/rwkv-lowrank-state-2026-11-23.MD (answer-count mix of 723 benchmark users); savings vs int8 from SM152 (card mean)
BYTES = {
    "Big": {"int8": 13891 + 9267, "t24": 2524 + 1720, "save": {"T24": 81.72, "F6": 84.60}, "store": (672.7, 185.0)},
    "Small": {"int8": 2545 + 1315, "t24": 839 + 470, "save": {"T24": 66.26, "F6": 71.44}, "store": (119.3, 51.9)},
    "Tiny": {"int8": 1315 + 1315, "t24": 462 + 470, "save": {"T24": 64.65, "F6": 69.74}, "store": (80.3, 44.9)},
}
LONG_CARDS = {  # copied from the release-candidate re-check (SM164) warnings
    "sm91s2": "with T24 the curve head is -0.016% [-0.092, +0.086] and the instant head +0.026% [-0.084, +0.110] vs its "
              "own int8 (SM155, 50 users); the 6-bit-factor version has no long-card read-out yet (the Big long-card "
              "check with 6-bit factors (SM165) is running).",
    "sm99": "with T24 the instant head is +0.006% [-0.105, +0.061] vs its own int8 (SM155, 50 users); the 6-bit-factor "
            "version has no long-card read-out yet (the Big long-card check with 6-bit factors (SM165) is running).",
    "sm118": "with 6-bit factors no cell is flagged (SM152, 50 users), and in every model of that check at most 0.107% of cards have a review whose "
             "recall moves by more than 0.10.",
    "sm113": "with 6-bit factors no cell is flagged (SM152, 50 users), and in every model of that check at most 0.107% of cards have a review whose "
             "recall moves by more than 0.10.",
    "sm130th": "with T24 the curve head is +0.002% and the instant head -0.044% vs its own int8 (SM148, 50 users); 6-bit "
               "factors were not taken for Tiny because their curve head on long cards reached a CI upper bound of "
               "+1.852% (SM152).",
    "sm130io": "with T24 the instant head is +0.009% vs its own int8 (SM148, 50 users).",
}
SPEED = {  # SM163, fork main eda153ddb, quiet machine, pairs, all threads, 205,785 synthetic reviews
    "Big": {"rps": "14,346", "vs_int8": "+38.8%", "rebuild": "14.6 s", "show_p50": "3.0 ms", "answer_p50": "0.76 ms"},
    "Small": {"rps": "45,156", "vs_int8": "-3.9%", "rebuild": "4.7 s", "show_p50": "0.64 ms", "answer_p50": "0.20 ms"},
    "Tiny": {"rps": "80,569", "vs_int8": "-3.9%", "rebuild": "2.7 s", "show_p50": "0.36 ms", "answer_p50": "0.11 ms"},
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fork_blob(rel):
    """Raw bytes of a fork main file (the git object, no checkout filters)."""
    return subprocess.run(["git", "-C", str(FORK), "cat-file", "blob", f"{FORK_COMMIT}:{FORK_INFER}/{rel}"],
                          check=True, capture_output=True).stdout


def pct(new, ref):
    return f"{100.0 * (new / ref - 1):+.2f}%"


def parity_line(pkg):
    p = json.loads((pkg / "parity.json").read_text(encoding="utf-8"))
    out = []
    for name, v in p.get("variants", {}).items():
        d = v.get("max_abs_dp_labelled", {})
        if d:
            out.append(f"{name} {max(d.values()):.1e}")
    return ", ".join(out) or "see parity.json"


def part_a(p, pkg, files, time_report):
    arch = json.loads((pkg / "architecture.json").read_text(encoding="utf-8"))
    man = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    head = json.loads((pkg / "time_head.json").read_text(encoding="utf-8"))
    fmt = FORMATS[p["fmt"]]
    b = BYTES[p["size"]]
    sp = SPEED[p["size"]]
    heads = list(p["logloss"])
    two = p["variant"] == "two-head"
    L = []
    L.append(f"# Integration manual: {p['size']} {p['variant']}, {p['name']} (package {p['key']}, release {TAG} draft)")
    L.append("")
    L.append(f"This manual has two parts. **Part A** (this part) is new for the {TAG} family release and describes "
             f"this package: the model, its scores, the state format the Anki fork stores for it, its answer-time "
             f"head and its test vectors. **Part B** is the full porting manual of {p['pred']}, kept verbatim: file "
             f"formats, the 92 input features, the leak-free input rule (section 6.9), the show-time intervals "
             f"(section 6.10), the forward pass, the state, the outputs and the port plan. **Where the two differ, "
             f"Part A wins**: Part B's model-specific numbers (accuracy, hashes, parameter counts of another model, "
             f"test-vector rows) belong to the model it was written for.")
    L.append("")
    L.append("## A.1 What it is")
    L.append("")
    L.append(f"- Model: {p['name']}, scorer label `{p['label']}`, package key `{p['key']}`; {p['size']} size, "
             f"{p['variant']} model of the RWKV family.")
    L.append(f"- Architecture `{arch['architecture']}`, {man['parameters']:,} parameters, layer schedule "
             f"`{arch['layer_schedule']}`.")
    L.append(f"- Heads to use: {p['heads']}.")
    L.append(f"- Inputs: leak-free (Part B section 6.9) and show-time intervals (Part B section 6.10), as for every "
             f"model of the family.")
    L.append(f"- {p['what']}")
    if p.get("six_layer"):
        streams = ", ".join(f"{s['name']} {s['n_layers']}" for s in arch["streams"])
        L.append(f"- **Six-layer architecture.** Layers per stream: {streams}; `layer_steps` "
                 f"{json.dumps(arch['layer_steps'])} (six steps in two layer rounds instead of the twelve steps of "
                 f"the one-card-layer Tiny (`w64-c1`) that Part B describes). Widths, heads, features and the "
                 f"card and note streams are unchanged (one card layer, one note layer), so the stored card and note "
                 f"state is the same size as the Tiny two-head model (SM106)'s. Where Part B lists tensors (section "
                 f"4.3), shows `architecture.json` (section 5) or walks the layer schedule (section 7), this "
                 f"package's `architecture.json` and the NumPy reference in `reference/` are authoritative: the "
                 f"reference reads the schedule and the layer counts from `architecture.json`, and the test vectors "
                 f"check it.")
    L.append("")
    L.append("## A.2 Accuracy")
    L.append("")
    L.append("Held-out benchmark users, 732-user set (722 users), review-weighted LogLoss, show-time inputs, leak-free, "
             "int8 weights and int8 state (w8+s8). \"% vs big\" is against the big model (1 leak-free epoch) on the "
             "same inputs (curve 0.29958, instant 0.277157), as in the 13 November tables; negative is better.")
    L.append("")
    L.append("| Head | LogLoss (w8+s8) | % vs big | Cost of the fork's state format vs own int8 (732 set, 95% CI) | "
             "Release budget check (SM164, CI upper bound) |")
    L.append("|---|---|---|---|---|")
    for h in heads:
        L.append(f"| {h} | {p['logloss'][h]:.6f} | {pct(p['logloss'][h], BIG_REF[h])} | {p['sm164'][h]} | "
                 f"{p['budget'][h]}: PASS |")
    if not two:
        L.append("")
        L.append("The curve head is untrained; its LogLoss is not meaningful and not listed.")
    L.append("")
    L.append("The budgets (user decisions of 2026-11-23): Big at most +0.15% vs its own int8 state; Small at most +0.5% "
             "and Tiny at most +1.0% in total vs the matching Big model at int8, judged by the 95% CI upper bound per "
             "head on the 732 set and on the 1,500-user pool (release-candidate re-check of the final state formats "
             "(SM164), all three sizes GO).")
    L.append("")
    L.append("## A.3 State format in the Anki fork")
    L.append("")
    L.append(f"The fork (main {FORK_COMMIT[:9]}, profile value `compact_lowrank`, the default) stores this model's card "
             f"and note state as **T24 low-rank states with {fmt['factors']}**: precision byte `{fmt['byte']}` under "
             f"the magic `ARWKVMODSTATE2`, the training repository's label `{fmt['label']}`. Each WKV matrix is "
             f"stored as a rank-2 product below 16 answers and a rank-4 product from the 16th answer (sum of L_i "
             f"R_i^T per head); the x and channel shift vectors stay int8. The answer's own output uses the float "
             f"update; only the stored state is approximated. Big and Small use int6 factors (`0x26`), the six-layer "
             f"Tiny int8 factors (`0x28`): each size runs only its own models, so a pair never mixes widths. The "
             f"full byte layout and the write rule are in the fork's `docs/rwkv-lowrank-state-2026-11-23.MD` and "
             f"`rslib/src/rwkv/lowrank.rs`; Part B section 8 describes the int8 rows (`s8`) that this format "
             f"replaces, and remains the accuracy reference.")
    L.append("")
    own = b["save"][p["fmt"]]
    est = b["int8"] * (1 - own / 100)
    L.append(f"| {p['size']} size, one model | int8 rows | T24, int8 factors | **this package's format ({p['fmt']})** |")
    L.append("|---|---|---|---|")
    L.append(f"| card + note state (fork layout, mean over the answer counts of 723 benchmark users) | {b['int8']:,} B "
             f"| {b['t24']:,} B | about {est:,.0f} B |")
    L.append(f"| saving vs int8 rows (scorer, card mean, SM152) | 0 | -{b['save']['T24']:.1f}% | **-{own:.1f}%** |")
    L.append(f"| state store of a synthetic 14,000-card collection, both models of the size (SM163) | "
             f"{b['store'][0]} MB | | {b['store'][1]} MB |")
    L.append("")
    L.append(f"Long cards (128 or more answers before the review) are checked as a warning, not a rule: "
             f"{LONG_CARDS[p['key']]} The format was applied after training (no retrain).")
    L.append("")
    L.append("## A.4 Answer-time head")
    L.append("")
    L.append(f"The fork runs an answer-time head beside this model (Card Info: answer time by button, the chance of "
             f"answering within 2 to 30 s). Files: `time_head.f32.bin` (`ARWKVWEIGHTS1`), `time_head.int8.bin` "
             f"(`ARWKVWEIGHTS8PC1`, int8 per output row) and `time_head.json` (format_version "
             f"{head['format_version']}, the complete spec: inputs, layout, bins, valid rows).")
    L.append("")
    L.append(f"- Head: logits = out(relu(hidden(inputs))), {head['weights']['tensors']['time_mlps.0.hidden.weight'][1]}"
             f" inputs -> 128 -> 64 logits (4 grades x 16 log-time bins, softmax per grade). Inputs = "
             f"{head['inputs']['frozen']['count']} frozen trunk values at the query (instant) row (the "
             f"layer-normalised residual after the encoder and after each "
             f"{'stream' if p['size'] == 'Big' else 'layer round'}, plus the instant head's hidden layer) and 26 "
             f"explicit history features (`time_head.json` \"inputs\").")
    L.append("- The answer time is the whole time on the card, front and back: question shown to grade press, capped "
             "at the preset's maximum answer seconds. Valid on the query row of every review that is not the card's "
             "first; same-day reviews included.")
    L.append("- The head never changes the model: the trunk it reads is this package's weights, bit for bit.")
    if p["key"] == "sm91s2":
        L.append("- In the fork the files sit beside the big weights as `RWKV_trained_on_5000_10000.time_head.*`; this "
                 "package names them `time_head.*` like the others. `time_head.json` is byte-identical to the fork's "
                 "and its \"weights\" field still gives the fork's file names.")
    L.append("- In the fork a head always runs on the model it was trained on: one model alone uses its own head; a pair "
             "uses its Instant-Only model's head. The fork's release notes give the answer-time LogLoss as 17% to "
             "18% lower than a per-button running-median baseline for every size.")
    L.append("- NumPy reference: `reference/answer_time_numpy.py`; check: `python testvectors/check_time_vectors.py`.")
    L.append("")
    L.append("## A.5 Files and byte identity with the Anki fork")
    L.append("")
    L.append("| File | Bytes | sha256 | In fork main |")
    L.append("|---|---|---|---|")
    for f in files:
        L.append(f"| `{f['path']}` | {f['bytes']:,} | `{f['sha256']}` | {f['fork']} |")
    L.append("")
    note = [f"Fork main {FORK_COMMIT[:9]} bundles every file marked identical above (compared byte for byte "
            f"with the git objects)."]
    if p["size"] == "Big":
        note.append("`weights.int8.bin` is not in the fork (its Big engine runs the f32 weights); it is the same file "
                    "as in the 13 November release.")
    if p["key"] == "sm91s2":
        note.append("The fork keeps no `architecture.json` or `manifest.json` for this model (the big engine's layout "
                    "is built in); `master.pth` is the fork's `RWKV_trained_on_5000_10000.pth`.")
    else:
        note.append("`architecture.json` and `manifest.json` differ from the fork's copies only in line endings "
                    "(architecture.json) and in provenance fields (manifest.json: the git commit and the synthetic "
                    "test-vector notes).")
    L.append(" ".join(note))
    L.append("")
    L.append(f"Parity (NumPy reference vs PyTorch, `parity.json`, max abs error on labelled rows): {parity_line(pkg)}.")
    L.append("")
    L.append("## A.6 Test vectors (synthetic only)")
    L.append("")
    L.append("`testvectors/user_900001.npz` and `user_900002.npz` are two synthetic users: simulated review histories "
             "(fixed seed) in the dataset's on-disk format, written by `scripts/synthetic_testvector_users.py` of the "
             "training repository. No row of the training data is in this package. `time_user_900001.npz` and "
             "`time_user_900002.npz` hold the answer-time rows of the same users (frozen inputs for the first 500 "
             "time rows, explicit features, logits, the revlog columns the features come from).")
    L.append("")
    L.append("```")
    L.append("python testvectors/check_testvectors.py     # model: fp32 + int8, recurrent + parallel form; RESULT: PASS")
    L.append("python testvectors/check_time_vectors.py    # answer-time head: f32 + int8 head;            RESULT: PASS")
    L.append("```")
    L.append("")
    tr = time_report
    L.append(f"Both need only NumPy and run from any folder. The answer-time vectors were built with PyTorch on the CPU "
             f"from this package's `master.pth` and the fork's head files, and the logits of the rows that the fork's "
             f"own parity fixture (`rslib/src/rwkv/testdata/*-time-vectors.json.gz`) holds equal the fixture's "
             f"(largest difference {tr['fixture_max_abs_logit_diff']:.1e}).")
    L.append("")
    L.append("## A.7 Speed in the fork (SM163)")
    L.append("")
    L.append(f"Quiet desktop, fork main {FORK_COMMIT[:9]}, release build, the size's pair, all threads, a synthetic "
             f"collection of 14,000 cards and 205,785 answers: first-launch state build {sp['rps']} reviews/s "
             f"({sp['vs_int8']} vs int8 rows), whole rebuild {sp['rebuild']}; per review p50 {sp['show_p50']} at "
             f"show time (four button intervals included) and {sp['answer_p50']} for an answer.")
    L.append("")
    L.append("## A.8 License")
    L.append("")
    L.append("The model weights are not under the MIT License of the code: read `WEIGHTS-NOTICE.md` (the dataset's "
             "research terms; the open redistribution question, srs-benchmark issue #353). The answer-time head files "
             "(`time_head.*.bin`) are trained weights too and fall under the same notice. The test vectors come from "
             "synthetic users; no dataset rows are shipped.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"# Part B: the porting manual of {p['pred']} (verbatim)")
    L.append("")
    return "\n".join(L) + "\n"


def write_zip(src, out):
    """One top-level folder named after the zip; sorted entries, fixed timestamps (reproducible bytes)."""
    name = out.stem
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d in sorted({p.parent for p in src.rglob("*") if p.is_file()} | {src}):
            rel = d.relative_to(src)
            arc = f"{name}/" + ("" if str(rel) == "." else rel.as_posix() + "/")
            info = zipfile.ZipInfo(arc, ZIP_TIME)
            info.external_attr = 0o40755 << 16
            z.writestr(info, b"")
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f"{name}/{f.relative_to(src).as_posix()}", ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, f.read_bytes(), compresslevel=9)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "work/release-2026-11-28"))
    args = ap.parse_args()
    out = Path(args.out)
    fork_main = out / "fork-main"
    pkg_root, dist = out / "pkg", out / "dist"
    shutil.rmtree(pkg_root, ignore_errors=True)
    shutil.rmtree(dist, ignore_errors=True)
    pkg_root.mkdir(parents=True)
    dist.mkdir(parents=True)
    notice = fork_blob("WEIGHTS-NOTICE.md")
    if notice != (HF / "WEIGHTS-NOTICE.md").read_bytes():
        raise SystemExit("WEIGHTS-NOTICE.md: fork main and the last release differ")
    old = b"the `weights.*.bin` and `master.pth` files"
    if notice.count(old) != 1:
        raise SystemExit("WEIGHTS-NOTICE.md: expected sentence not found exactly once")
    # SM166 packages also ship the answer-time head weights: name them (nothing else changes)
    notice = notice.replace(old, b"the `weights.*.bin`, `time_head.*.bin` and `master.pth` files")
    manifest = {"tag_proposed": TAG, "published": False, "fork_main": FORK_COMMIT, "packages": {}, "assets": {}}
    for p in PACKAGES:
        pkg = pkg_root / p["zip"]
        src = Path(p["src"])
        for rel in COPY:
            dst = pkg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, dst)
        extra = sorted(str(f.relative_to(src)).replace("\\", "/") for f in src.rglob("*") if f.is_file()
                       and str(f.relative_to(src)).replace("\\", "/") not in COPY)
        # synthetic users only: no other user_*.npz may exist in the source testvectors
        users = sorted(f.name for f in (src / "testvectors").glob("*.npz"))
        if users != ["user_900001.npz", "user_900002.npz"]:
            raise SystemExit(f"{p['zip']}: unexpected test vectors {users}")
        (pkg / "WEIGHTS-NOTICE.md").write_bytes(notice)
        for ext in ("f32.bin", "int8.bin", "json"):
            data = fork_blob(f"{p['fork_head']}.{ext}")
            staged = (fork_main / p["key"] / f"time_head.{ext}").read_bytes()
            if data != staged:
                raise SystemExit(f"{p['zip']}: staged head file differs from fork main")
            (pkg / f"time_head.{ext}").write_bytes(data)
        shutil.copyfile(HERE / "package_files/reference/answer_time_numpy.py", pkg / "reference/answer_time_numpy.py")
        shutil.copyfile(HERE / "package_files/testvectors/check_time_vectors.py",
                        pkg / "testvectors/check_time_vectors.py")
        tv = out / "time-vectors" / p["key"]
        time_report = json.loads((tv / "time-vectors-report.json").read_text(encoding="utf-8"))
        if time_report["result"] != "PASS":
            raise SystemExit(f"{p['zip']}: time vectors did not pass")
        for u in (900001, 900002):
            shutil.copyfile(tv / f"time_user_{u}.npz", pkg / f"testvectors/time_user_{u}.npz")
        # byte identity with fork main
        fork_map = {"weights.f32.bin": p["fork_weights"], "weights.int8.bin": p["fork_int8"],
                    "master.pth": p["fork_master"], "time_head.f32.bin": f"{p['fork_head']}.f32.bin",
                    "time_head.int8.bin": f"{p['fork_head']}.int8.bin", "time_head.json": f"{p['fork_head']}.json"}
        files = []
        for rel in ("weights.f32.bin", "weights.int8.bin", "master.pth", "time_head.f32.bin", "time_head.int8.bin",
                    "time_head.json"):
            data = (pkg / rel).read_bytes()
            f = {"path": rel, "bytes": len(data), "sha256": sha256_bytes(data)}
            if fork_map[rel]:
                fb = fork_blob(fork_map[rel])
                f["fork_path"] = f"{FORK_INFER}/{fork_map[rel]}"
                f["fork_identical"] = fb == data
                f["fork"] = f"identical: `{fork_map[rel]}`" if fb == data else f"DIFFERS from `{fork_map[rel]}`"
                if fb != data:
                    raise SystemExit(f"{p['zip']} {rel}: differs from fork main {fork_map[rel]}")
            else:
                f["fork_identical"] = None
                f["fork"] = ("not in the fork (no int8 Big weights; the Big engine runs f32)" if rel == "weights.int8.bin"
                             else "not shipped by the fork (the fork's build bundles weights.f32.bin)")
            files.append(f)
        text = part_a(p, pkg, files, time_report) + p["base"].read_text(encoding="utf-8")
        (pkg / "INTEGRATION.md").write_text(text, encoding="utf-8", newline="\n")
        listing = {str(f.relative_to(pkg)).replace("\\", "/"): {"bytes": f.stat().st_size, "sha256": sha256(f)}
                   for f in sorted(pkg.rglob("*")) if f.is_file()}
        bad = [k for k in listing if k.endswith((".safetensors", ".log")) or "__pycache__" in k]
        if bad:
            raise SystemExit(f"{p['zip']}: unexpected files {bad}")
        manifest["packages"][p["zip"]] = {
            "model": p["name"], "key": p["key"], "label": p["label"], "size": p["size"], "variant": p["variant"],
            "source": str(src).replace("\\", "/"), "source_files_left_out": extra,
            "integration_part_b": str(p["base"].relative_to(ROOT)).replace("\\", "/"),
            "state_format": {"name": p["fmt"], **FORMATS[p["fmt"]]}, "fork_files": files,
            "time_vectors": time_report, "files": listing}
        write_zip(pkg, dist / f"{p['zip']}.zip")
        print(f"[sm166] {p['zip']}.zip {(dist / (p['zip'] + '.zip')).stat().st_size:,} B "
              f"({len(listing)} files; weights identical to fork main)", flush=True)
    shutil.copyfile(pkg_root / "rwkv-small-two-head/INTEGRATION.md", dist / "INTEGRATION.md")
    (dist / "WEIGHTS-NOTICE.md").write_bytes(notice)
    names = [f"{p['zip']}.zip" for p in PACKAGES] + ["INTEGRATION.md", "WEIGHTS-NOTICE.md"]
    lines = [f"{sha256(dist / n)}  {n}" for n in names]
    (dist / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    for n in names + ["SHA256SUMS"]:
        manifest["assets"][n] = {"bytes": (dist / n).stat().st_size, "sha256": sha256(dist / n)}
    (HERE / "release-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    shutil.copyfile(dist / "SHA256SUMS", HERE / "SHA256SUMS")
    print((dist / "SHA256SUMS").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
