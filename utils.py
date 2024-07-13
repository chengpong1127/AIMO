from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    Pipeline,
)
import torch
import gc


def get_pipe(model_path: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def generate_completions(
    pipe: Pipeline, prompt: str, batch_size: int, generate_kwargs: dict = {}
) -> list[str]:
    prompts = [prompt] * batch_size
    clean_memory()
    generated_texts = pipe(prompts, return_full_text=False, **generate_kwargs)
    generated_texts = [text[0]["generated_text"] for text in generated_texts]

    return generated_texts
