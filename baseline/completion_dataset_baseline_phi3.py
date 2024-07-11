# %% [markdown]
# # Stage2: Math dataset completion

# %%
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
def get_pipe(model_path: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quantization_config, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

# %%
from dataset_generator import generate_completion_dataset
from dataset import get_MATH_dataset
import re

model_path = "microsoft/Phi-3-medium-128k-instruct"
completion_prompt = "{problem} \nPlease reason step by step, and put your final answer within \\boxed{{}}.\nApproach: "
dataset_save_path = "dataset/completion_dataset_MATH_LLAMA3_8b_ZeroShot_COT"

dataset = get_MATH_dataset("test")
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

# %%
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


