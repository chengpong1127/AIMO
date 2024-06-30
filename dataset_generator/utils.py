import torch
import gc
from transformers import Pipeline



def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
    
def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False
    
def generate_completions(pipe: Pipeline, prompt: str, batch_size: int, generate_kwargs: dict = {}) -> list[str]:
    prompts = [prompt] * batch_size
    clean_memory()
    generated_texts = pipe(prompts, return_full_text=False, **generate_kwargs)
    generated_texts = [text[0]['generated_text'] for text in generated_texts]
    
    return generated_texts
    