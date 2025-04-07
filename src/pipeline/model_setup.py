from transformers import AutoModelForSequenceClassification, PreTrainedModel
from config.pipeline_config import ModelConfig
import typing as tp


def load_model(
    config: ModelConfig, label2id: tp.Mapping[str, int], id2label: tp.Mapping[int, str]
) -> PreTrainedModel:
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification",
    )
    return model
