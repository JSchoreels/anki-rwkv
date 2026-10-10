"""FSRS feature selection used solely to align RWKV benchmark labels."""

from config import Config
from .fsrs_engineer import FSRSFeatureEngineer


def create_feature_engineer(config: Config) -> FSRSFeatureEngineer:
    if config.model_name not in {"FSRS-5", "FSRS-6", "FSRS-7"}:
        raise ValueError("RWKV label alignment supports FSRS-5, FSRS-6, and FSRS-7")
    return FSRSFeatureEngineer(config)
