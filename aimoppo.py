
import yaml
import re
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, load_tool, BitsAndBytesConfig, Trainer
from tqdm.auto import tqdm

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import bitsandbytes as bnb

load_dotenv()

@dataclass
class TrainerConfig:
    model_name: str
    prompt_path: str
    save_dir: str
    wandb_project_name: str
    
    # Text Environment
    text_env_tools: Optional[list[str]] = None
    text_env_max_turns: int = 1
    
    # Training
    epochs: int = 4
    train_batch_size: int = 4
    train_mini_batch_size: int = 1
    val_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    generation_max_new_tokens: int = 512
    
    

class AIMOPPOTrainer:
    def __init__(self, config: TrainerConfig, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            peft_config=lora_config,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        )
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        with open(self.config.prompt_path, "r") as f:
            prompt = f.read()
        
        generation_kwargs = {
            "max_new_tokens": self.config.generation_max_new_tokens,
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": -1,
        }
        self.text_env = TextEnvironment(
            self.model,
            self.tokenizer,
            [load_tool(tool) for tool in self.config.text_env_tools],
            self._exact_match_reward,
            prompt,
            max_turns=self.config.text_env_max_turns,
            generation_kwargs=generation_kwargs,
        )
        ppo_config = PPOConfig(
            batch_size=self.config.train_batch_size,
            mini_batch_size=self.config.train_mini_batch_size,
            ppo_epochs=self.config.epochs,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            remove_unused_columns=False,
            optimize_cuda_cache=True,
            log_with="wandb",
            tracker_project_name=self.config.wandb_project_name,
            project_kwargs = {
                "project_dir": self.config.save_dir,
                "automatic_checkpoint_naming": True,
                "total_limit": 5,
            },
            accelerator_kwargs = {
                "mixed_precision": "bf16"
            }
        )
        adam = bnb.optim.Adam8bit(self.model.parameters())
        self.ppo_trainer: PPOTrainer = PPOTrainer(config=ppo_config, model=self.model, tokenizer=self.tokenizer, dataset=train_dataset, optimizer=adam)
        self.model = self.ppo_trainer.model
        self.train_dataloader = self.ppo_trainer.dataloader
        self.val_dataloader = self.ppo_trainer.prepare_dataloader(val_dataset)
        
        self.accelerator = self.ppo_trainer.accelerator
        
    def _train_step(self, batch, text_env: TextEnvironment, ppo_trainer: PPOTrainer):
        queries, responses, masks, rewards, _ = text_env.run(batch["query"], answers=batch["answer"])
        train_stats = ppo_trainer.step(queries, responses, rewards, masks)
        return train_stats, queries, responses, rewards
    
    def _evaluate(self, test_dataloader, text_env: TextEnvironment, ppo_trainer: PPOTrainer):
        test_rewards = []
        for test_batch in test_dataloader:
            _, _, _, rewards, histories = text_env.run(test_batch["query"], answers=test_batch["answer"])
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
        self._train_inner_loop(self.ppo_trainer, self.text_env, self.tokenizer, self.config.epochs, self.train_dataloader, self.val_dataloader)
        reward_mean_test = self._evaluate(self.val_dataloader, self.text_env, self.ppo_trainer)
        self._log({"val_reward_mean": reward_mean_test})
        self.save_checkpoint()
                
    def _train_inner_loop(self, ppo_trainer: PPOTrainer, text_env: TextEnvironment, tokenizer, epochs, train_dataloader, val_dataloader):
        
        progress_bar = tqdm(range(epochs * len(train_dataloader)))
        self.current_step = 0
        for _ in range(epochs):
            for step, batch in enumerate(train_dataloader):
                if step == 0:
                    reward_mean_test = self._evaluate(val_dataloader, text_env, ppo_trainer)
                    self._log({"val_reward_mean": reward_mean_test})
                    self.save_checkpoint()

                train_stats, _, responses, rewards = self._train_step(batch, text_env, ppo_trainer)
                responses = [tokenizer.decode(response) for response in responses]
                
                self._log_stats(ppo_trainer, batch["query"], responses, batch["answer"], rewards, train_stats)
                self.current_step += 1
                progress_bar.update(1)
                
                
                
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
        
    def save_model(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        
    def infer(self, math_problem: str):
        return self._get_answer(self.infer_text(math_problem))
    
    def infer_text(self, math_problem: str):
        _, _, _, _, histories = self.text_env.run([math_problem], answers=[0])
        return histories[0].text
    
    def generate_aimo_submission(self):
        import aimo
        env = aimo.make_env()
        iter_test = env.iter_test()
        for test, sample_submission in iter_test:
            sample_submission['answer'] = self.infer(test['problem'])
            env.predict(sample_submission)
            print(test)
            print(sample_submission, '\n')
        