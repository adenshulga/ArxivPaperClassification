from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation and preprocessing"""

    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42


@dataclass
class CustomTrainingArguments:
    output_dir: str = "data/checkpoints"
    overwrite_output_dir: bool = True
    num_train_epochs: float = 3
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    # lr_scheduler_kwargs={},
    warmup_ratio: float = 0.03125
    warmup_steps: int = 1
    # per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    log_level: str = "error"
    # logging_dir="output_dir/runs/CURRENT_DATETIME_HOSTNAME"  # логи для tensorboard (default)
    logging_strategy: str = "steps"
    logging_steps: int = 1
    save_strategy: str = "best"
    # save_steps=1,
    save_total_limit: int = 2
    save_safetensors: bool = True  # safetensors вместо torch.save / torch.load
    save_only_model: bool = True  # сохраняем optimizer, shceduler, rng, ...
    use_cpu: bool = False
    seed: int = 42
    # bf16=True,  # использовать bf16 вместо fp32
    eval_strategy: str = "epoch"
    # eval_steps=32,
    disable_tqdm: bool = False
    load_best_model_at_end: bool = True
    label_smoothing_factor: float = 0.0
    optim: str = "adamw_torch"
    # optim_args=...,
    # resume_from_checkpoint: str = "last-checkpoint"
    auto_find_batch_size: bool = True
    report_to: str = "comet_ml"
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True


@dataclass
class ModelConfig:
    """Configuration for model architecture and parameters"""

    model_name: str = "allenai/scibert_scivocab_uncased"

    # model_name: tp.Literal[
    #     "FacebookAI/roberta-base",
    #     "distilbert-base-uncased",
    #     "allenai/scibert_scivocab_uncased",
    # ] = "allenai/scibert_scivocab_uncased"


@dataclass
class PipelineConfig:
    """Master configuration for the entire pipeline"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: CustomTrainingArguments = field(default_factory=CustomTrainingArguments)


cs = ConfigStore.instance()
cs.store(name="pipeline_config", node=PipelineConfig)
