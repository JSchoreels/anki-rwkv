"""Run the current accepted-path dispatch diagnostic under R01 preflight guards."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.profile_r01_dispatch import main

if __name__ == "__main__":
    main()
