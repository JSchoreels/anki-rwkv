"""CPU regression check for process exits during R01 memory telemetry."""

from pathlib import Path
import sys
from types import SimpleNamespace
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gpu_r01_training import sampled_tree_rss


class Child:
    def __init__(self, value):
        self.value = value

    def memory_info(self):
        if isinstance(self.value, Exception):
            raise self.value
        return SimpleNamespace(rss=self.value)


class Parent(Child):
    def children(self, recursive):
        assert recursive
        return [Child(2**20), Child(psutil.NoSuchProcess(123)), Child(psutil.AccessDenied(456))]


assert sampled_tree_rss(Parent(2 * 2**20)) == (3, 2)
print('Passed RSS sampling: surviving processes counted; exited and inaccessible children skipped')
