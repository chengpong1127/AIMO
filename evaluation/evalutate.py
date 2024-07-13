from transformers import Pipeline
from datasets import Dataset
from tqdm.auto import tqdm
from utils import clean_memory, is_number

def evaluate(pipe: Pipeline, dataset: Dataset, prompt: str, extract_answer: callable, batch_size: int = 1, generate_kargs: dict = {}, return_history = False):
    if 'problem' not in dataset.column_names or 'answer' not in dataset.column_names:
        raise ValueError("Dataset must contain 'problem' and 'answer' columns")
    
    history = []
    results = []
    current_accuracy = 0.0
    
    progress_bar = tqdm(dataset.iter(batch_size), total=len(dataset)//batch_size, desc=f"Evaluating accuracy. Current accuracy: {current_accuracy:.2f}")
    for batch in progress_bar:
        prompts = [prompt.format(problem=problem) for problem in batch['problem']]
        clean_memory()
        completions = pipe(prompts, return_full_text=False, **generate_kargs)
        generated_texts = [completion[0]["generated_text"] for completion in completions]
        generated_answers = [extract_answer(text) for text in generated_texts]
        results.extend([float(answer) == float(generated_answer) if is_number(answer) and is_number(generated_answer) else False for answer, generated_answer in zip(batch['answer'], generated_answers)])
        history.extend(list(zip(generated_texts, results)))
        current_accuracy = sum(results) / len(results)
        progress_bar.set_description(f"Evaluating accuracy. Current accuracy: {current_accuracy:.2f}")
        
    if return_history:
        return current_accuracy, history
    return current_accuracy