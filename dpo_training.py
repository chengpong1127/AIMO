# %%
from datasets import load_from_disk, load_dataset
from dataset_generator import generate_dpo_dataset
from trl import DPOConfig, DPOTrainer
import os
os.environ["WANDB_PROJECT"] = 'CARS'
os.environ["WANDB_NOTEBOOK_NAME"] = "dpo_training.ipynb"

dataset_path = "altas-ai/corrective_dataset_MATH_LLAMA3_8b_ZeroShot_COT"
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
#prompt = "{problem} \nPlease reason step by step, and put your final answer within \\boxed{{}}.\nApproach: "
dataset = load_dataset(dataset_path)
split = dataset['train'].train_test_split(test_size=0.2)

train_dataset = split['train']
eval_dataset = split['test']

# Rename columns
train_dataset = train_dataset.rename_column("original_prompt", "prompt")
train_dataset = train_dataset.rename_column("incorrect_completion", "rejected")
train_dataset = train_dataset.rename_column("correct_completion", "chosen")

eval_dataset = eval_dataset.rename_column("original_prompt", "prompt")
eval_dataset = eval_dataset.rename_column("incorrect_completion", "rejected")
eval_dataset = eval_dataset.rename_column("correct_completion", "chosen")

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
    #target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                  "gate_proj", "up_proj", "down_proj",],
    #target_modules=["q_proj", "v_proj"],
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

        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        beta=0.1,
        
        gradient_accumulation_steps=4, 
        eval_accumulation_steps=4,
        gradient_checkpointing=True,
        
        max_length=1536,
        max_prompt_length=512,
        max_target_length=1024,
        remove_unused_columns=False,
        truncation_mode="keep_start",
        
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="wandb",
        run_name="DPO_2",
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


