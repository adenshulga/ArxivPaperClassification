from transformers import pipeline, Pipeline
import typing as tp
from config.inference_config import InferenceConfig
import streamlit as st

from src.app.tags_mapping import tags2full_name


class LabelScore(tp.TypedDict):
    label: str
    score: float


@st.cache_resource
def setup_pipeline(cfg: InferenceConfig) -> Pipeline:
    model = pipeline(
        "text-classification", model=cfg.checkpoint_path, tokenizer=cfg.model_name
    )
    return model


def get_top_labels(scores: list[LabelScore], top_percent: float) -> list[LabelScore]:
    top_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    cumulative_score = 0
    selected_labels: list[LabelScore] = []
    for score in top_scores:
        cumulative_score += score["score"]
        selected_labels.append(score)
        if cumulative_score >= top_percent:
            break
    return selected_labels


def get_full_names(
    labels: list[LabelScore], label2name: dict[str, str]
) -> list[LabelScore]:
    return [
        LabelScore(label=label2name[label["label"]], score=label["score"])
        if label["label"] in label2name
        else LabelScore(label=label["label"], score=label["score"])
        for label in labels
    ]


def get_top_label_names(
    scores: list[LabelScore], label2name: dict[str, str], top_percent: float
) -> list[LabelScore]:
    top_labels = get_top_labels(scores, top_percent)
    return get_full_names(top_labels, label2name)
