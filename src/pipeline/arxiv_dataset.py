import typing as tp
from collections.abc import Mapping

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from config.pipeline_config import DatasetConfig
from loguru import logger
import ast
import string


class ArxivTag(tp.TypedDict):
    term: str
    scheme: str


class ArxivPaper(tp.TypedDict):
    summary: str
    tag: str
    title: str


def load_arxiv_dataset() -> Dataset:
    df = pd.read_json("data/arxivData.json").head(100)
    dataset = Dataset.from_pandas(df[["summary", "tag", "title"]])
    return dataset


def multihot_encode_tags(
    tags: list[str], label_mapping: Mapping[str, int]
) -> list[float]:
    num_labels = len(label_mapping)
    labels = [0.0] * num_labels

    for tag in tags:
        if tag in label_mapping:
            labels[label_mapping[tag]] = 1.0

    return labels


def generate_preprocessing_function(
    tokenizer: PreTrainedTokenizerBase,
    label_mapping: Mapping[str, int],
):
    def preprocess_arxiv_dataset(row: ArxivPaper):
        text = row["title"] + " " + row["summary"]

        tokenized_text = tokenizer(
            text,
            truncation=True,
            padding="max_length",
        )

        tags_list: list[ArxivTag] = ast.literal_eval(row["tag"])

        tags = [tag["term"] for tag in tags_list]
        tokenized_text["labels"] = multihot_encode_tags(tags, label_mapping)

        return tokenized_text

    return preprocess_arxiv_dataset


def get_label_mappings(dataset: Dataset) -> tuple[dict[str, int], dict[int, str]]:
    """
    Create label mappings from dataset

    Args:
        dataset: Dataset containing tag information

    Returns:
        tuple containing:
            - label2id: Mapping from label names to indices
            - id2label: Mapping from indices to label names
    """
    all_tags = set()
    for idx in range(len(dataset)):
        example = dataset[idx]
        tags_list: list[ArxivTag] = ast.literal_eval(example["tag"])
        for tag in tags_list:
            if not any(digit in tag["term"] for digit in string.digits):
                all_tags.add(tag["term"])

    logger.info(f"Found {len(all_tags)} unique tags")

    label2id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    id2label = {idx: tag for tag, idx in label2id.items()}

    return label2id, id2label


def train_val_test_split(
    dataset: Dataset, cfg: DatasetConfig
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation and test sets

    Args:
        dataset: Dataset to split
        cfg: Dataset configuration containing split parameters

    Returns:
        tuple containing train, validation and test datasets
    """
    shuffled_dataset = dataset.shuffle(seed=cfg.seed)

    train_temp = shuffled_dataset.train_test_split(
        test_size=cfg.val_size + cfg.test_size
    )

    remaining_test_ratio = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_test = train_temp["test"].train_test_split(test_size=remaining_test_ratio)

    train_dataset = train_temp["train"]
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    logger.info(
        f"Split dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def prepare_arxiv_dataset(
    tokenizer: PreTrainedTokenizerBase,
    cfg: DatasetConfig,
    dataset: Dataset,
    label2id: dict[str, int],
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load arxiv dataset, preprocess it and return train/validation/test split

    Args:
        tokenizer: Tokenizer to use for text processing
        cfg: Dataset configuration
        dataset: Optional pre-loaded dataset, will load if None
        label2id: Optional pre-computed label mapping, will create if None

    Returns:
        tuple containing:
            - train dataset
            - validation dataset
            - test dataset
    """

    preprocess_fn = generate_preprocessing_function(
        tokenizer=tokenizer,
        label_mapping=label2id,
    )

    processed_dataset = dataset.map(
        preprocess_fn, remove_columns=["summary", "tag", "title"]
    )

    return train_val_test_split(processed_dataset, cfg)


def invert_label_mapping(label_mapping: Mapping[str, int]) -> dict[int, str]:
    return {idx: label for label, idx in label_mapping.items()}
