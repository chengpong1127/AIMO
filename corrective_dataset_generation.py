from datasets import load_from_disk
from dataset_generator import generate_corrective_dataset
from transformers import StoppingCriteria, StoppingCriteriaList
from utils import get_pipe
import re

completion_dataset_path = "dataset/completion_dataset_Phi-3-mini-128k-instruct_ZeroShot_COT"
model_path = "mmicrosoft/Phi-3-mini-128k-instruct"
dataset_save_path = "dataset/corrective_dataset_MATH_Phi-3-mini-128k-instruct_ZeroShot_COT"
corrective_prompt = """{problem}
Here is the incorrect solution generated from LLM.
{incorrect_solution}, The answer we got from the incorrect solution is {incorrect_answer}. Note that if the incorrect answer is None, it means the model did not generate answer following the format.
Here is the correct answer of this problem: {correct_answer}.
You are a smart and talented math professor, you need to correct the incorrect solution that can lead to the correct answer.
Please reason step by step, and put your final answer within \\boxed{{}}.
Approach: """

def get_answer_from_output(text):
    try:
        result_output = re.findall(r"\\boxed\{(\d+)\}", text)
        return float(result_output[-1])
    except Exception:
        return None

class StoppingCriteriaSub(StoppingCriteria):
    def __call__(self, input_ids, scores):
        decoded_text = pipe.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        decoded_text = decoded_text[-20:]
        return re.search(r"\\boxed\{(.+)\}", decoded_text) is not None

generate_kargs = {
    "max_new_tokens": 2048,
    "do_sample": True, 
    "batch_size": 4,
    "top_k": 0.0,
    "top_p": 1.0,
    "temperature": 0.5,
    "stopping_criteria" : StoppingCriteriaList([StoppingCriteriaSub()]),
}

pipe = get_pipe(model_path)
completion_dataset = load_from_disk(completion_dataset_path)
incorrect_dataset = completion_dataset.filter(lambda x: x["label"] is False)


corrective_dataset, history = generate_corrective_dataset(
    incorrect_solution_dataset=incorrect_dataset,
    corrective_prompt=corrective_prompt,
    supervisor_pipe=pipe,
    extract_answer_function=get_answer_from_output,
    generate_kwargs=generate_kargs,
    corrective_solution_count_per_incorrect_solution=1,
    correction_attempts=5,
    return_completion_history=True,
    checkpoint_path="_".join([dataset_save_path, "checkpoint"]),
)
corrective_dataset.save_to_disk(dataset_save_path)