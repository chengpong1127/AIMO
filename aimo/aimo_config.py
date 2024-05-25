from dataclasses import dataclass, field

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
    gradient_accumulation_steps: int = 4
    generation_max_new_tokens: int = 512
    learning_rate: float = 1e-5
    
    precision: str = "bf16"