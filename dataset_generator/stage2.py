from transformers import Pipeline
from datasets import Dataset
from tqdm.auto import tqdm
from .utils import is_number, generate_completions



    
def generate_completion_dataset(pipe: Pipeline, dataset: Dataset, prompt: str, extract_answer_function: callable, generate_count_per_problem: int = 5, generate_kwargs: dict = {}) -> Dataset:
    result_prompts = []
    result_completions = []
    result_labels = []
    result_problem = []
    result_correct_answers = []
    result_generated_answers = []
    
    for row in tqdm(dataset.iter(1), total=len(dataset)):
        prompt_with_problem = prompt.format(problem=row["problem"][0])
        generated_text = generate_completions(pipe, prompt_with_problem, generate_count_per_problem, generate_kwargs)
        generated_answers = [extract_answer_function(completion) for completion in generated_text]
        
        result_prompts += [prompt_with_problem] * generate_count_per_problem
        result_completions += generated_text
        result_labels += [float(generated_answer) == float(row["answer"][0]) if is_number(generated_answer) else False for generated_answer in generated_answers]
        result_problem += [row["problem"][0]] * generate_count_per_problem
        result_correct_answers += row["answer"] * generate_count_per_problem
        result_generated_answers += generated_answers
        
        
    return Dataset.from_dict({
        "prompt": result_prompts,
        "completion": result_completions,
        "label": result_labels,
        "problem": result_problem,
        "correct_answer": result_correct_answers,
        "generated_answer": result_generated_answers
    })