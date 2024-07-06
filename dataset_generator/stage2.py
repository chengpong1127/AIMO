from transformers import Pipeline
from datasets import Dataset
from tqdm.auto import tqdm
from .utils import is_number, generate_completions


def generate_completion_dataset(pipe: Pipeline, dataset: Dataset, prompt: str, extract_answer_function: callable, generate_count_per_problem: int = 1, generate_kwargs: dict = {}) -> Dataset:
    result_prompts = []
    result_completions = []
    result_labels = []
    result_problem = []
    result_correct_answers = []
    result_generated_answers = []

    # Use batching for efficiency
    batch_size = 32  # Adjust based on your GPU memory and model size
    dataset = list(dataset)  # Convert to list for indexing

    for i in tqdm(range(0, len(dataset), batch_size), total=(len(dataset) // batch_size) + 1):
        batch = dataset[i:i + batch_size]
        prompts = [prompt.format(problem=row["question"][0]) for row in batch]
        all_generated_text = []

        for prompt_with_problem in prompts:
            generated_text = generate_completions(pipe, prompt_with_problem, generate_count_per_problem, generate_kwargs)
            all_generated_text.append(generated_text)
        
        for row, generated_texts in zip(batch, all_generated_text):
            generated_answers = [extract_answer_function(completion) for completion in generated_texts]
            
            result_prompts += [prompt.format(problem=row["question"][0])] * generate_count_per_problem
            result_completions += generated_texts
            result_labels += [
                float(generated_answer) == float(row["answer"][0]) if generated_answer is not None and is_number(row["answer"][0]) else False
                for generated_answer in generated_answers
            ]
            result_problem += [row["question"][0]] * generate_count_per_problem
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