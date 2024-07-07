import re
from transformers import StoppingCriteria, StoppingCriteriaList
from dataset_generator import generate_completion_dataset
from dataset import get_MATH_dataset
from utils import get_pipe

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
completion_prompt = "{problem} \nPlease reason step by step, and put your final answer within \\boxed{{}}.\nApproach: "
dataset_save_path = "dataset/completion_dataset_MATH_LLAMA3_8b_ZeroShot_COT"

dataset = get_MATH_dataset()
pipe = get_pipe(model_path)
class StoppingCriteriaSub(StoppingCriteria):
    def __call__(self, input_ids, scores):
        decoded_text = pipe.tokenizer.decode(input_ids[0], skip_special_tokens=True)
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
def get_answer_from_output(text):
    try:
        result_output = re.findall(r"\\boxed\{(\d+)\}", text)
        return float(result_output[0])
    except Exception:
        return None
    
    
completion_dataset = generate_completion_dataset(
    pipe=pipe,
    dataset=dataset,
    prompt=completion_prompt,
    extract_answer_function=get_answer_from_output,
    generate_kwargs=generate_kargs,
    generate_count_per_problem=1,
    checkpoint_path="_".join([dataset_save_path, "checkpoint"]),
)
completion_dataset.save_to_disk(dataset_save_path)