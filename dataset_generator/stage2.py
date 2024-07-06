from transformers import Pipeline
from datasets import Dataset, load_from_disk, disable_progress_bar, DatasetInfo
from tqdm.auto import tqdm
from .utils import is_number, generate_completions

disable_progress_bar()

def generate_completion_dataset(
    pipe: Pipeline,
    dataset: Dataset,
    prompt: str,
    extract_answer_function: callable,
    generate_count_per_problem: int = 1,
    generate_kwargs: dict = {},
    checkpoint_path: str = None,
) -> Dataset:  
    result_prompts = []
    result_completions = []
    result_labels = []
    result_problem = []
    result_correct_answers = []
    result_generated_answers = []
    start_index = 0
    if checkpoint_path is not None:
        try:
            checkpoint_dataset = load_from_disk(checkpoint_path)
            result_prompts = checkpoint_dataset["prompt"]
            result_completions = checkpoint_dataset["completion"]
            result_labels = checkpoint_dataset["label"]
            result_problem = checkpoint_dataset["problem"]
            result_correct_answers = checkpoint_dataset["correct_answer"]
            result_generated_answers = checkpoint_dataset["generated_answer"]
            start_index = int(checkpoint_dataset.info.description) + 1
            print(f"Resuming from index {start_index}")
        except FileNotFoundError:
            print("Checkpoint file not found. Starting from scratch.")

    for i in tqdm(range(start_index, len(dataset)), desc="Generating completions"):
        row = dataset[i]
        prompt_with_problem = prompt.format(problem=row["problem"])
        generated_text = generate_completions(
            pipe, prompt_with_problem, generate_count_per_problem, generate_kwargs
        )
        generated_answers = [
            extract_answer_function(completion) for completion in generated_text
        ]

        result_prompts += [prompt_with_problem] * generate_count_per_problem
        result_completions += generated_text
        result_labels += [
            (
                float(generated_answer) == float(row["answer"])
                if is_number(generated_answer)
                else False
            )
            for generated_answer in generated_answers
        ]
        result_problem += [row["problem"]] * generate_count_per_problem
        result_correct_answers += [row["answer"]] * generate_count_per_problem
        result_generated_answers += generated_answers
        
        if checkpoint_path is not None:
            checkpoint_dataset = Dataset.from_dict(
            {
                "prompt": result_prompts,
                "completion": result_completions,
                "label": result_labels,
                "problem": result_problem,
                "correct_answer": result_correct_answers,
                "generated_answer": result_generated_answers,
            },
            info=DatasetInfo(str(i))
            )
            checkpoint_dataset.save_to_disk(checkpoint_path)

    return Dataset.from_dict(
        {
            "prompt": result_prompts,
            "completion": result_completions,
            "label": result_labels,
            "problem": result_problem,
            "correct_answer": result_correct_answers,
            "generated_answer": result_generated_answers,
        },
        info=DatasetInfo(
            description=f"Completion generation dataset.\nPrompt:{prompt}\nmodel_path:{pipe.model.name_or_path}\ngenerate_count_per_problem:{generate_count_per_problem}\ngenerate_kwargs:{str(generate_kwargs)}"
        )
    )
