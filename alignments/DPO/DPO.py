from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

class DPO:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype='bfloat16',
        )

    def train(self, train_dataset: Dataset, test_dataset: Dataset, epochs: int):
        trainer = DPOTrainer(
            model=self.model,
            args=DPOConfig(
                output_dir='output',
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                bf16=True,
                gradient_accumulation_steps=4,
                do_eval=True,
                logging_strategy='steps',
                save_strategy='steps',
                evaluation_strategy='steps',
                logging_steps=100,
                save_steps=100,
                eval_steps=100,
                num_train_epochs=epochs,
                load_best_model_at_end=True,
                warmup_steps=200,
            ),
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            peft_config=self.lora_config,
        )
        trainer.train()
        trainer.save_model("model_output")