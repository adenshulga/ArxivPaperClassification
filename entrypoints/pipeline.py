import comet_ml  # noqa: F401
import hydra
from transformers import AutoTokenizer, Trainer, TrainingArguments
from loguru import logger
from src.pipeline.arxiv_dataset import (
    get_label_mappings,
    load_arxiv_dataset,
    prepare_arxiv_dataset,
)
from config.pipeline_config import PipelineConfig
from src.pipeline.metrics import compute_metrics
from src.pipeline.model_setup import load_model
from src.pipeline.env_setup import setup_env
from src.pipeline.logging_setup import setup_logging
from transformers.integrations import CometCallback
from transformers import DataCollatorWithPadding


@hydra.main(config_name="pipeline_config", version_base="1.2")
def main(cfg: PipelineConfig):
    logger.info("Setting up environment variables")
    setup_env()
    logger.info("Setting up logging")
    experiment = setup_logging()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    logger.info("Loading dataset")
    dataset = load_arxiv_dataset()
    label2id, id2label = get_label_mappings(dataset)

    train_dataset, val_dataset, test_dataset = prepare_arxiv_dataset(
        tokenizer=tokenizer,
        cfg=cfg.dataset,
        dataset=dataset,
        label2id=label2id,
    )
    logger.info("Loading model")
    model = load_model(cfg.model, label2id, id2label)
    logger.info("Training model")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**cfg.training),  # type: ignore
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[CometCallback()],
    )

    trainer.train()
    logger.info("Evaluating model")
    results = trainer.evaluate(test_dataset)  # type: ignore
    logger.info(results)
    experiment.log_metrics(results)
    experiment.end()


if __name__ == "__main__":
    main()
