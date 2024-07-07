import os
import re
from datasets import load_dataset, Dataset
from typing import Literal

## The dataset subpackage contains functions to get all datasets used in the project. Datasets are formatted in the Huggingface Dataset format.


_AIMO_VAL_PATH = os.path.join(os.path.dirname(__file__), 'AIMO_val.csv')
_MATH_PATH = os.path.join(os.path.dirname(__file__), 'MATH')

def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False
    
def _get_answer_from_solution(solution):
    try:
        return re.search(r'\\boxed\{(\d+)\}', solution).group(1)
    except AttributeError:
        return None

def get_AIMO_val_dataset(max_problem_length=512) -> Dataset:
    dataset = Dataset.from_csv(_AIMO_VAL_PATH)
    dataset = dataset.select_columns(['problem', 'answer'])
    dataset = dataset.filter(lambda x: _is_number(x['answer']))
    dataset = dataset.filter(lambda x: len(x['problem']) <= max_problem_length)
    return dataset

def get_MATH_dataset(dataset_type: Literal['train', 'test'] = 'train', max_problem_length=512) -> Dataset:
    dataset = load_dataset("json", data_dir=_MATH_PATH)
    dataset = dataset[dataset_type]
    dataset = dataset.filter(lambda x: len(x['problem']) <= max_problem_length)
    dataset = dataset.map(lambda x: {'problem': x['problem'], 'answer': _get_answer_from_solution(x['solution'])})
    dataset = dataset.filter(lambda x: _is_number(x['answer']))
    dataset = dataset.select_columns(['problem', 'answer'])
    return dataset

def get_GSM8k_dataset(dataset_type: Literal['train', 'test'] = 'train', max_problem_length=512) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset[dataset_type]
    dataset = dataset.rename_column('question', 'problem')
    dataset = dataset.filter(lambda x: len(x['problem']) <= max_problem_length)
    dataset = dataset.map(lambda x: {'problem': x['problem'], 'answer': x['answer'].split('####')[1].strip()})
    dataset = dataset.filter(lambda x: _is_number(x['answer']))
    return dataset


if __name__ == '__main__':
    print(get_GSM8k_dataset())