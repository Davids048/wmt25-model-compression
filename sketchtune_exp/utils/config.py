# config.py with a single dataclass representing the whole structure
from transformers import HfArgumentParser, SchedulerType
from dataclasses import dataclass, field
from typing import Tuple
import datetime
import os

# Define nested dataclasses
@dataclass
class ModelConfig:
    model_path: str
    model_quantizer_path: str 


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int 
    per_device_eval_batch_size: int 
    num_train_epochs: int 
    gradient_accumulation_steps: int 
    num_warmup_steps: int
    learning_rate: float
    weight_decay: float
    grad_clip: float=1.0
    seed: int=1234
    lr_scheduler_type: SchedulerType = SchedulerType.COSINE
    max_seq_len: int=2048 
    eval_step: int=500
    eval_delay: int=0


@dataclass
class DataConfig:
    data_path: str 
    source_lang: str 
    target_lang: str
    val_set_size: int
    

def create_exp_output_dir(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    data_config: DataConfig,
) -> str :
    """Create a output directory for an experiment run
    
    Convention: 
    - All runs reside in the `exp_runs` dir. 
    - Upper level dirs: 
        - base model 
        - {source}-{target} pair 
    - Runs are identifyed as (SUBJECT TO CHANGE) 
        {quantizer_info}-{eff_batch_size}-{lr}-{timestamp}
    """
    parent_dir = f"./exp_runs/{data_config.source_lang}-{data_config.target_lang}"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    model_info = os.path.splitext(os.path.basename(model_config.model_quantizer_path))[0]
    eff_batch_sz = train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps
    timestamp = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    output_dir = f"{model_info}/BS{eff_batch_sz}-LR{train_config.learning_rate}/{timestamp}"
    full_dir = os.path.join(parent_dir,output_dir)

    if not os.path.exists(full_dir):
        os.makedirs(full_dir) 
    else:
        raise FileExistsError()

    assert os.path.exists(full_dir), "Failed to create exp output dir!"
    return full_dir






def parse_exp_args_yaml(yaml:str) -> Tuple[ModelConfig, TrainingConfig, DataConfig]:
    parser = HfArgumentParser((ModelConfig, TrainingConfig, DataConfig))
    model_config, training_config, data_config = parser.parse_yaml_file(yaml)

    assert isinstance(model_config, ModelConfig)
    assert isinstance(training_config, TrainingConfig)
    assert isinstance(data_config, DataConfig)

    if isinstance(training_config.learning_rate, str):
        training_config.learning_rate = float(training_config.learning_rate)

    return model_config, training_config, data_config


