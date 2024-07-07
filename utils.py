
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline


def get_pipe(model_path: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe