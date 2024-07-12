# %%
from datasets import load_from_disk
from dataset_generator import generate_dpo_dataset
from trl import DPOConfig, DPOTrainer
import os
os.environ["WANDB_PROJECT"] = 'CAMR'
os.environ["WANDB_NOTEBOOK_NAME"] = "dpo_training.ipynb"

dataset_path = "dataset/corrective_dataset_MATH_Phi-3-mini-128k-instruct_ZeroShot_COT"
model_path = "microsoft/Phi-3-mini-128k-instruct"



dataset = load_from_disk(dataset_path)
dataset = generate_dpo_dataset(dataset)

split = dataset.train_test_split(test_size=0.1)

train_dataset = split['train']
eval_dataset = split['test']

# %%
from unsloth import FastLanguageModel
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    load_in_4bit = True,
    dtype=torch.bfloat16
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
)

# %%


dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args=DPOConfig(
        output_dir="models",
        
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=1,
        
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        
        gradient_accumulation_steps=4, 
        eval_accumulation_steps=4,
        gradient_checkpointing=True,
        
        max_length=1300,
        max_prompt_length=300,
        max_target_length=1000,
        remove_unused_columns=False,
        truncation_mode="keep_start",
        
        load_best_model_at_end=True,
        save_total_limit=3,
        
        report_to="wandb",
        run_name="DPO",
        bf16 = True,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# %%
dpo_trainer.train()

# %%
import wandb
wandb.finish()


