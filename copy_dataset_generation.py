import re
from transformers import StoppingCriteria, StoppingCriteriaList
from dataset_generator import generate_copy_dataset
from datasets import load_from_disk
from utils import get_pipe

corrective_dataset_path = "dataset/corrective_dataset_MATH_LLAMA3_8b_ZeroShot_COT"
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_save_path = "dataset/corrective_copy_dataset_MATH_LLAMA3_8b_ZeroShot_COT"
copy_prompt = """
Problem: {problem}
Correct Solution: {correct_solution}
Incorrect Solution: {incorrect_solution}
<End>
This is the problem and the correct and incorrect solution generated by LLM. Please generate one similar problem with a similar correct solution and a similar incorrect solution in the same format.
Do not generate any other unnecessary information and only generate these three parts: problem, correct solution, and incorrect solution, Put <End> at the end of the output.

Problem: """


pipe = get_pipe(model_path)

    
def extract_prompt_correct_incorrect(text):
    pattern = re.compile(
        r"\s*(?P<problem>.+?)\n"
        r"Correct Solution:\s*(?P<correct_solution>.+?)\n"
        r"Incorrect Solution:\s*(?P<incorrect_solution>.+?)\n<End>",
        re.DOTALL,
    )

    match = pattern.search(text)
    if match:
        return (
            match.group("problem"),
            match.group("correct_solution"),
            match.group("incorrect_solution"),
        )
    else:
        return None

class StoppingCriteriaSub(StoppingCriteria):
    def __call__(self, input_ids, scores):
        decoded_text = pipe.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return decoded_text.endswith("<End>")
    
generate_kargs = {
    "max_new_tokens": 2048,
    "do_sample": True, 
    "batch_size": 4,
    "top_k": 0.0,
    "top_p": 1.0,
    "temperature": 0.5,
    "stopping_criteria" : StoppingCriteriaList([StoppingCriteriaSub()]),
}



corrective_dataset = load_from_disk(corrective_dataset_path)


copied_dataset = generate_copy_dataset(
    dataset=corrective_dataset,
    copy_prompt=copy_prompt,
    pipe=pipe,
    copy_count=1,
    extract_problem_correct_incorrect=extract_prompt_correct_incorrect,
    generate_kwargs=generate_kargs,
    checkpoint_path="_".join([dataset_save_path, "checkpoint"])
)
copied_dataset.save_to_disk(dataset_save_path)