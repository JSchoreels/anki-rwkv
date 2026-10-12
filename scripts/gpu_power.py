"""Monitor NVIDIA power; optionally apply and restore a supported limit."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import threading
import time


def query_gpu(index, fields):
    result = subprocess.run(
        ["nvidia-smi", "-i", str(index), f"--query-gpu={','.join(fields)}",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    values = result.stdout.strip().split(", ")
    if len(values) != len(fields):
        raise RuntimeError(f"Unexpected nvidia-smi response: {result.stdout.strip()}")
    return dict(zip(fields, values))


def set_power_limit(index, watts):
    result = subprocess.run(
        ["nvidia-smi", "-i", str(index), "-pl", str(watts)],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode:
        raise RuntimeError(f"Could not set GPU power limit: {result.stdout.strip()} {result.stderr.strip()}")
    actual = float(query_gpu(index, ["power.limit"])["power.limit"])
    if abs(actual - watts) > 0.5:
        raise RuntimeError(f"Requested {watts} W, but the GPU reports {actual} W")


@contextmanager
def limited_gpu(index, watts, output):
    info = query_gpu(index, ["uuid", "name", "power.min_limit", "power.max_limit", "power.limit"])
    minimum, maximum = float(info["power.min_limit"]), float(info["power.max_limit"])
    original = float(info["power.limit"])
    if watts and not minimum <= watts <= maximum:
        raise ValueError(f"{info['name']} accepts {minimum:g}-{maximum:g} W; requested {watts:g} W")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    errors = []
    readings = []
    started = time.monotonic()
    info.update(requested_watts=watts, original_watts=original, restored=False)

    base_fields = ["power.draw", "power.limit", "temperature.gpu", "utilization.gpu", "memory.used"]
    # Clocks, P-state and clock-event reasons separate a slow GPU state from slower code.
    state_fields = ["clocks.sm", "clocks.mem", "pstate", "clocks_event_reasons.active"]
    fields = base_fields + state_fields

    def parse(name, value):
        if name == "pstate":
            return int(value.lstrip("P"))
        if name == "clocks_event_reasons.active":
            return int(value, 16)
        return float(value)

    def monitor():
        nonlocal fields
        with (output / "gpu_power.jsonl").open("a", encoding="utf-8") as log:
            while not stop.is_set():
                try:
                    try:
                        reading = query_gpu(index, fields)
                    except (RuntimeError, subprocess.CalledProcessError):
                        if fields == base_fields:
                            raise
                        errors.append(f"GPU state fields unavailable; monitoring {', '.join(base_fields)} only")
                        fields = base_fields
                        reading = query_gpu(index, fields)
                    reading = {name: parse(name, value) for name, value in reading.items()}
                    reading["elapsed_seconds"] = time.monotonic() - started
                    readings.append(reading["power.draw"])
                    log.write(json.dumps(reading) + "\n")
                    log.flush()
                except Exception as exc:
                    errors.append(str(exc))
                stop.wait(1)

    worker = None
    changed = False
    try:
        # Mark before the command: a driver command can succeed but its verification fail.
        if watts:
            changed = True
            set_power_limit(index, watts)
            print(f"GPU power cap: {watts:g} W (previously {original:g} W)", flush=True)
        else:
            print(f"Monitoring GPU power; leaving the {original:g} W cap unchanged", flush=True)
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        yield info
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=12)
        try:
            if changed:
                actual = float(query_gpu(index, ["power.limit"])["power.limit"])
                if abs(actual - original) > 0.5:
                    set_power_limit(index, original)
                info["restored"] = True
                print(f"Restored GPU power cap to {original:g} W", flush=True)
            info["power_limit_changed"] = changed
        finally:
            info.update(samples=len(readings), monitor_errors=errors,
                        mean_power_watts=sum(readings) / len(readings) if readings else None,
                        max_sampled_power_watts=max(readings) if readings else None)
            (output / "power_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
