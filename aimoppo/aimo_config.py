from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TrainerConfig:
    model_name: str = ""
    prompt: str = ""
    prompt_path: str = ""
    save_dir: str = ""
    wandb_project_name: str = ""
    
    # Text Environment
    text_env_tools: list[str] = field(default_factory=list)
    text_env_max_turns: int = 1
    
    # Training
    epochs: int = 4
    train_batch_size: int = 4
    train_mini_batch_size: int = 1
    val_batch_size: int = 4
    test_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    generation_max_new_tokens: int = 512
    save_steps: int = 20
    learning_rate: float = 0.00001
    warnup_steps: int = 10
    optimizer: Literal["adam", "adamw", "adam8b", "adamw8b"] = "adamw8b"
    lr_scheduler: Literal["constant", "linear", "cosine"] = "constant"
    
    precision: str = "bf16"
    