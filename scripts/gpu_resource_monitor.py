"""Low-rate process-tree/CPU/memory telemetry for complete training comparisons."""

from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

import psutil


class ResourceMonitor:
    def __init__(self, folder, interval=1.0):
        self.folder, self.interval = Path(folder), interval
        self.phase = "startup/primer"
        self.stop = threading.Event()
        self.samples = 0
        self.sampling_seconds = 0.0
        self.errors = []
        self.unavailable = {}
        self.sampling_thread_cpu_seconds = 0.0
        self.processes = {}

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        psutil.cpu_percent(percpu=True)
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def collect(self):
        parent = psutil.Process()
        with (self.folder / "resources.jsonl").open("x", encoding="utf-8") as file:
            while not self.stop.is_set():
                started = time.perf_counter()
                cpu_started = time.thread_time()
                try:
                    processes = []
                    for process in [parent, *parent.children(recursive=True)]:
                        try:
                            if process.pid not in self.processes:
                                self.processes[process.pid] = process
                                process.cpu_percent()
                                cpu = None
                            else:
                                process = self.processes[process.pid]
                                cpu = process.cpu_percent()
                            memory = process.memory_info()
                            io = process.io_counters()
                            processes.append({"pid": process.pid, "parent_pid": process.ppid(), "name": process.name(),
                                              "cpu_percent_one_core_100": cpu, "rss_bytes": memory.rss,
                                              "page_faults": getattr(memory, "num_page_faults", None),
                                              "io_read_bytes": io.read_bytes, "io_write_bytes": io.write_bytes})
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    memory = psutil.virtual_memory()
                    # Windows paging counters can be disabled while CPU/RAM/I/O
                    # counters remain usable. Retain the rest of each sample.
                    swap = None
                    if "swap/paging" not in self.unavailable:
                        try:
                            swap = psutil.swap_memory()
                        except Exception as exc:
                            self.unavailable["swap/paging"] = str(exc)
                    sample = {"elapsed_seconds": time.perf_counter() - self.started, "phase": self.phase,
                              "cpu_per_core_percent": psutil.cpu_percent(percpu=True), "processes": processes,
                              "process_tree_rss_bytes": sum(p["rss_bytes"] for p in processes),
                              "ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
                              "ram_used_percent": memory.percent, "swap_used_bytes": swap.used if swap else None,
                              "swap_in_bytes": swap.sin if swap else None, "swap_out_bytes": swap.sout if swap else None}
                    if "GPU utilization query" not in self.unavailable:
                        try:
                            from scripts.gpu_power import query_gpu
                            values = query_gpu(0, ["utilization.gpu", "utilization.memory", "memory.used"])
                            for counter, value in values.items():
                                try:
                                    sample["nvidia_" + counter] = float(value)
                                except ValueError:
                                    sample["nvidia_" + counter] = None
                                    self.unavailable[counter] = value
                        except Exception as exc:
                            self.unavailable["GPU utilization query"] = str(exc)
                    sample["unavailable_counters"] = dict(self.unavailable)
                    import torch
                    sample["torch_cuda_allocated_bytes"] = torch.cuda.memory_allocated() if torch.cuda.is_initialized() else None
                    sample["torch_cuda_reserved_bytes"] = torch.cuda.memory_reserved() if torch.cuda.is_initialized() else None
                    sample["torch_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None
                    file.write(json.dumps(sample, allow_nan=False) + "\n")
                    file.flush()
                    self.samples += 1
                except Exception as exc:
                    self.errors.append(str(exc))
                self.sampling_seconds += time.perf_counter() - started
                self.sampling_thread_cpu_seconds += time.thread_time() - cpu_started
                self.stop.wait(self.interval)

    @contextmanager
    def training_phase(self, original, *args):
        self.phase = "training including user transitions"
        try:
            with original(*args) as information:
                yield information
        finally:
            self.phase = "checkpoint save/evaluation"

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        result = {"interval_seconds": self.interval, "samples": self.samples,
                  "sampling_wall_seconds": self.sampling_seconds,
                  "sampling_thread_cpu_seconds": self.sampling_thread_cpu_seconds,
                  "case_wall_seconds": time.perf_counter() - self.started,
                  "sampling_wall_fraction": self.sampling_seconds / max(time.perf_counter() - self.started, 1e-9),
                  "sampling_errors": self.errors,
                  "scope": "one-second CPU/process-tree/RAM/I-O/paging and CUDA allocator samples; phases mark wrapper boundaries, user transition/queue timing in updates.jsonl",
                  "gpu_telemetry": "gpu_power.jsonl: existing one-second power, compute utilization, temperature and memory-used monitor during training",
                  "unavailable_counters": self.unavailable,
                  "unavailable": ["GPU byte/s bandwidth", "hardware kernel occupancy", "continuous per-worker peak RAM"],
                  "GPU_memory_activity": "nvidia utilization.memory is sampled memory activity percent where available; not achieved bytes/s bandwidth",
                  "overhead": "measured sampler wall time/fraction; no claim that this equals critical-path slowdown; all matched cases use the same telemetry"}
        (self.folder / "resources-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
