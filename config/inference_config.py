from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class InferenceConfig:
    """Configuration for inference"""

    model_name: str = "allenai/scibert_scivocab_uncased"
    checkpoint_path: str = "data/checkpoints/checkpoint-10"
    top_percent: float = 0.95
