from datasets import Dataset, DatasetInfo, disable_progress_bar, enable_progress_bar, load_from_disk
from transformers import Pipeline
from tqdm.auto import tqdm
from utils import generate_completions, is_number
import logging
import yaml

def generate_completion_dataset(
    pipe: Pipeline,
    dataset: Dataset,
    prompt: str,
    extract_answer_function: callable,
    generate_count_per_problem: int = 1,
    generate_kwargs: dict = {},
    checkpoint_path: str = None,
) -> Dataset:  
    disable_progress_bar()
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
    enable_progress_bar()
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


def generate_corrective_dataset(
    incorrect_solution_dataset: Dataset,
    corrective_prompt: str,
    supervisor_pipe: Pipeline,
    extract_answer_function: callable,
    corrective_solution_count_per_incorrect_solution: int = 1,
    correction_attempts=5,
    generate_kwargs={},
    return_completion_history=False,
    checkpoint_path: str = None,
) -> Dataset:
    disable_progress_bar()
    result_original_prompts = []
    result_incorrect_solutions = []
    result_correction_prompts = []
    result_corrective_completions = []
    start_index = 0
    if return_completion_history:
        completion_history = []
    if checkpoint_path is not None:
        try:
            checkpoint_dataset = Dataset.load_from_disk(checkpoint_path)
            result_original_prompts = checkpoint_dataset["original_prompt"]
            result_incorrect_solutions = checkpoint_dataset["incorrect_completion"]
            result_correction_prompts = checkpoint_dataset["correction_prompt"]
            result_corrective_completions = checkpoint_dataset["correct_completion"]
            start_index = int(checkpoint_dataset.info.description) + 1
            if return_completion_history:
                with open(f"{checkpoint_path}/history.yaml", "r") as f:
                    completion_history = yaml.load(f, Loader=yaml.FullLoader)
            logging.info(f"Resuming from index {start_index}")
        except FileNotFoundError:
            logging.info("Checkpoint file not found. Starting from scratch.")
    for i in tqdm(range(start_index, len(incorrect_solution_dataset)), desc="Generating corrective completions"):
        row = incorrect_solution_dataset[i]
        prompt_with_problem = corrective_prompt.format(
            problem=row["problem"],
            incorrect_solution=row["completion"],
            correct_answer=row["correct_answer"],
            incorrect_answer=row["generated_answer"],
        )

        correct_completion = []
        attempt = 0
        while (
            len(correct_completion) < corrective_solution_count_per_incorrect_solution
            and attempt < correction_attempts
        ):
            completions = generate_completions(
                supervisor_pipe,
                prompt_with_problem,
                corrective_solution_count_per_incorrect_solution,
                generate_kwargs,
            )
            correct_completion += [
                completion
                for completion in completions
                if extract_answer_function(completion) == float(row["correct_answer"])
            ]
            attempt += 1
            if return_completion_history:
                completion_history += completions

        new_data_count = len(correct_completion)
        result_original_prompts += [row["prompt"]] * new_data_count
        result_incorrect_solutions += [row["completion"]] * new_data_count
        result_correction_prompts += [prompt_with_problem] * new_data_count
        result_corrective_completions += correct_completion[:new_data_count]
        
        if checkpoint_path is not None:
            checkpoint_dataset = Dataset.from_dict(
                {
                    "original_prompt": result_original_prompts,
                    "incorrect_completion": result_incorrect_solutions,
                    "correction_prompt": result_correction_prompts,
                    "correct_completion": result_corrective_completions,
                },
                info=DatasetInfo(str(i))
            )
            checkpoint_dataset.save_to_disk(checkpoint_path)
            if return_completion_history:
                with open(f"{checkpoint_path}/history.yaml", "w") as f:
                    yaml.dump(completion_history, f)
    enable_progress_bar()

    if return_completion_history:
        return (
            Dataset.from_dict(
                {
                    "original_prompt": result_original_prompts,
                    "incorrect_completion": result_incorrect_solutions,
                    "correction_prompt": result_correction_prompts,
                    "correct_completion": result_corrective_completions,
                },
                info=DatasetInfo(f"Corrective Dataset\nCorrective Prompt: {corrective_prompt}\nSupervisor Model Path: {supervisor_pipe.model.name_or_path}\nCorrection Attempts: {correction_attempts}\nCorrective Solution Count Per Incorrect Solution: {corrective_solution_count_per_incorrect_solution}\nGenerate kwargs: {generate_kwargs}"),
            ),
            completion_history,
        )
    else:
        return Dataset.from_dict(
            {
                "original_prompt": result_original_prompts,
                "incorrect_completion": result_incorrect_solutions,
                "correction_prompt": result_correction_prompts,
                "correct_completion": result_corrective_completions,
            },
            info=DatasetInfo(f"Corrective Dataset\nCorrective Prompt: {corrective_prompt}\nSupervisor Model Path: {supervisor_pipe.model.name_or_path}\nCorrection Attempts: {correction_attempts}\nCorrective Solution Count Per Incorrect Solution: {corrective_solution_count_per_incorrect_solution}\nGenerate kwargs: {generate_kwargs}"),
        )


def generate_copy_dataset(
    dataset: Dataset,
    copy_prompt: str,
    pipe: Pipeline,
    copy_count: int,
    extract_problem_correct_incorrect: callable,
    generate_kwargs={},
    checkpoint_path: str = None,
) -> Dataset:
    disable_progress_bar()
    result_problems = []
    result_correct_solutions = []
    result_incorrect_solutions = []
    start_index = 0
    if checkpoint_path is not None:
        try:
            checkpoint_dataset = Dataset.load_from_disk(checkpoint_path)
            result_problems = checkpoint_dataset["prompt"]
            result_correct_solutions = checkpoint_dataset["correct_completion"]
            result_incorrect_solutions = checkpoint_dataset["incorrect_completion"]
            start_index = int(checkpoint_dataset.info.description) + 1
            logging.info(f"Resuming from index {start_index}")
        except FileNotFoundError:
            logging.info("Checkpoint file not found. Starting from scratch.")

    for i in tqdm(range(start_index, len(dataset)), desc="Generating copy completions"):
        row = dataset[i]
        result_problems.append(row["original_prompt"])
        result_correct_solutions.append(row["correct_completion"])
        result_incorrect_solutions.append(row["incorrect_completion"])

        prompt = copy_prompt.format(
            problem=row["original_prompt"],
            correct_solution=row["correct_completion"],
            incorrect_solution=row["incorrect_completion"],
        )
        completions = generate_completions(pipe, prompt, copy_count, generate_kwargs)
        extract_result = [
            result for completion in completions
            if (result := extract_problem_correct_incorrect(completion)) is not None and len(result[0]) > 10 and len(result[1]) > 10 and len(result[2]) > 10
        ]
        if not extract_result:
            continue
        problems, correct_solutions, incorrect_solutions = zip(*extract_result)
        result_problems += problems
        result_correct_solutions += correct_solutions
        result_incorrect_solutions += incorrect_solutions
        
        if checkpoint_path is not None:
            checkpoint_dataset = Dataset.from_dict(
                {
                    "prompt": result_problems,
                    "correct_completion": result_correct_solutions,
                    "incorrect_completion": result_incorrect_solutions,
                },
                info=DatasetInfo(str(i))
            )
            checkpoint_dataset.save_to_disk(checkpoint_path)
    enable_progress_bar()

    return Dataset.from_dict(
        {
            "prompt": result_problems,
            "correct_completion": result_correct_solutions,
            "incorrect_completion": result_incorrect_solutions,
        },
        info=DatasetInfo(f"Copy Dataset\nCopy Prompt: {copy_prompt}\nSupervisor Model Path: {pipe.model.name_or_path}\nGenerate kwargs: {generate_kwargs}\nCopy Count: {copy_count}"),
    )