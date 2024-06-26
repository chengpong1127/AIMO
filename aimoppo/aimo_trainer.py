import yaml
import re
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, load_tool, BitsAndBytesConfig, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, Trainer
from tqdm.auto import tqdm

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment
from typing import Optional, Union
import bitsandbytes as bnb
from .aimo_config import TrainerConfig

from dotenv import load_dotenv
load_dotenv()



class AIMOPPOTrainer:
    def __init__(self, config: Union[TrainerConfig, str, dict], train_dataset: Optional[Dataset] = None, val_dataset: Optional[Dataset] = None, test_dataset: Optional[Dataset] = None, load_checkpoint: Optional[str] = None, inference: bool = False):
        if isinstance(config, str):
            with open(config, "r") as f:
                _config = yaml.safe_load(f)
            self.config = TrainerConfig(**_config)
        elif isinstance(config, dict):
            self.config = TrainerConfig(**config)
        elif isinstance(config, TrainerConfig):
            self.config = config
        else:
            raise ValueError("config must be a TrainerConfig, str for config path or dict")
        
        if self.config.precision == "bf16":
            self.precision = torch.bfloat16
        elif self.config.precision == "fp16":
            self.precision = torch.float16
        elif self.config.precision == "fp32":
            self.precision = torch.float32
        else:
            raise ValueError("precision must be one of bf16, fp16, fp32")
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.precision,
            bnb_4bit_quant_storage=self.precision,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            #target_modules="all-linear",
        )
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            peft_config=lora_config,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            torch_dtype=self.precision,
        )
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.config.prompt:
            prompt = self.config.prompt
        else:
            with open(self.config.prompt_path, "r") as f:
                prompt = f.read()
        
        self.text_env = TextEnvironment(
            self.model,
            self.tokenizer,
            [load_tool(tool) for tool in self.config.text_env_tools],
            self._exact_match_reward,
            prompt,
            max_turns=self.config.text_env_max_turns,
            generation_kwargs={
                "max_new_tokens": self.config.generation_max_new_tokens,
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": -1,
            },
        )
        ppo_config = PPOConfig(
            model_name=self.config.model_name,
            batch_size=self.config.train_batch_size,
            mini_batch_size=self.config.train_mini_batch_size,
            ppo_epochs=self.config.epochs,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            remove_unused_columns=False,
            optimize_cuda_cache=True,
            log_with="wandb" if not inference else None,
            tracker_project_name=self.config.wandb_project_name,
            learning_rate=self.config.learning_rate,
            project_kwargs = {
                "project_dir": self.config.save_dir,
                "automatic_checkpoint_naming": True,
                "total_limit": 5,
            },
            accelerator_kwargs = {
                "mixed_precision": self.config.precision
            }
        )
        optimizer = None
        if self.config.optimizer == "adam8b":
            optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "adamw8b":
            optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=self.config.learning_rate)
            
        if self.config.lr_scheduler == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warnup_steps)
        elif self.config.lr_scheduler == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warnup_steps, num_training_steps=464)
        elif self.config.lr_scheduler == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warnup_steps, num_training_steps=464)
        
        self.ppo_trainer: PPOTrainer = PPOTrainer(config=ppo_config, model=self.model, tokenizer=self.tokenizer, dataset=train_dataset, optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.model = self.ppo_trainer.model
        if train_dataset is not None:
            self.train_dataloader = self.ppo_trainer.dataloader
        if val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.val_batch_size,
                shuffle=True,
                drop_last=True,
            )
        if test_dataset is not None:
            self.test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.config.test_batch_size,
                shuffle=True,
                drop_last=True,
            )
        
        self.accelerator = self.ppo_trainer.accelerator
        
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
        
    def _train_step(self, batch, text_env: TextEnvironment, ppo_trainer: PPOTrainer):
        queries, responses, masks, rewards, _ = text_env.run(batch["query"], answers=batch["answer"])
        train_stats = ppo_trainer.step(queries, responses, rewards, masks)
        return train_stats, queries, responses, rewards
    
    def _evaluate(self, test_dataloader, text_env: TextEnvironment, ppo_trainer: PPOTrainer):
        test_rewards = []
        for test_batch in test_dataloader:
            _, _, _, rewards, _ = text_env.run(test_batch["query"], answers=test_batch["answer"])
            test_rewards.extend(rewards)
        test_rewards = ppo_trainer.accelerator.gather_for_metrics(
            torch.stack(test_rewards).to(ppo_trainer.accelerator.device)
        )
        return test_rewards.mean()
    
    def _exact_match_reward(self, responses, answers):
        """Reward if generated response contains correct answer."""
        rewards = []
        for response, answer in zip(responses, answers):
            reward = 0.0
            predicted_number = self._get_answer(response)
            if predicted_number is not None:
                if np.abs(predicted_number - float(answer)) < 0.1:
                    reward += 1.0
            rewards.append(torch.tensor(reward))
        return rewards
    
    def _get_answer(self, response):
        try:
            pattern = r"Result\s*=\s*(-?\d+(?:\.\d+)?)\s*<submit>"
            match_pattern = re.findall(pattern, response)
            if match_pattern:
                return float(match_pattern[0])
            else:
                return 0
        except Exception:
            return 0
    
    
    def train(self):
        if self.train_dataloader is None or self.val_dataloader is None:
            raise ValueError("train_dataset and val_dataset must be provided to train")
        self._train_inner_loop(self.ppo_trainer, self.text_env, self.tokenizer, self.config.epochs, self.train_dataloader, self.val_dataloader, self.test_dataloader)
                
    def _train_inner_loop(self, ppo_trainer: PPOTrainer, text_env: TextEnvironment, tokenizer, epochs, train_dataloader, val_dataloader, test_dataloader):
        
        progress_bar = tqdm(range(epochs * len(train_dataloader)))
        self.current_step = 0
        best_reward_mean = float('-inf')
        for _ in range(epochs):
            for _, batch in enumerate(train_dataloader):
                train_stats, _, responses, rewards = self._train_step(batch, text_env, ppo_trainer)
                responses = [tokenizer.decode(response) for response in responses]
                
                self._log_stats(ppo_trainer, batch["query"], responses, batch["answer"], rewards, train_stats)
                self.current_step += 1
                progress_bar.update(1)
                
                if self.current_step % self.config.save_steps == 0 and self.current_step > 0:
                    val_reward_mean = self._evaluate(val_dataloader, text_env, ppo_trainer)
                    test_reward_mean = self._evaluate(test_dataloader, text_env, ppo_trainer)
                    
                    self._log({"val_reward_mean": val_reward_mean})
                    self._log({"test_reward_mean": test_reward_mean})
                    if val_reward_mean > best_reward_mean:
                        best_reward_mean = val_reward_mean
                        self.save_checkpoint()
                
                
                
    def _log_stats(self, ppo_trainer: PPOTrainer, queries: list, responses: list, answers: list, rewards, train_stats: dict):
        texts = {
            "query": queries,
            "response": responses,
            "answer": answers,
        }
        ppo_trainer.log_stats(train_stats, texts, rewards, columns_to_log=["query", "response", "answer"])
        
    def _log(self, data: dict):
        self.accelerator.log(data, self.current_step)
        
    def save_checkpoint(self):
        self.accelerator.project_configuration.iteration = self.current_step
        self.accelerator.save_state()
    
    def load_checkpoint(self, checkpoint_dir: str):
        self.accelerator.load_state(checkpoint_dir)
        
    def infer_answer(self, math_problem: str, self_consistency_num: int = 1):
        answers = []
        for _ in range(self_consistency_num):
            _, responses, _, _, _ = self.text_env.run([math_problem], answers=[0])
            response = self.tokenizer.decode(responses[0])
            answers.append(self._get_answer(response))
        final_answer = max(set(answers), key=answers.count)
        return final_answer
        
    
    def infer(self, math_problem: str):
        _, _, _, _, histories = self.text_env.run([math_problem], answers=[0])
        return histories[0]