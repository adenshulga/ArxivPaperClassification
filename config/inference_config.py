from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for inference"""

    model_name: str = "allenai/scibert_scivocab_uncased"
    checkpoint_path: str = "data/checkpoints/checkpoint-12300"
    top_percent: float = 0.95
    minimal_score: float = 0.01


cfg = InferenceConfig()
