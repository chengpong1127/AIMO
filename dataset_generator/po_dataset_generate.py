from datasets import Dataset

def generate_dpo_dataset(dataset: Dataset):
    corrective_dataset_columns = ['original_prompt', 'incorrect_completion', 'correct_completion']
    copy_dataset_columns = ['prompt', 'incorrect_completion', 'correct_completion']
    
    if all(col in dataset.column_names for col in corrective_dataset_columns):
        return dataset.rename_columns({
            'original_prompt': 'prompt',
            'incorrect_completion': 'rejected',
            'correct_completion': 'chosen'
        }).select_columns(['prompt', 'rejected', 'chosen'])
    
    elif all(col in dataset.column_names for col in copy_dataset_columns):
        return dataset.rename_columns({
            'prompt': 'prompt',
            'incorrect_completion': 'rejected',
            'correct_completion': 'chosen'
        }).select_columns(['prompt', 'rejected', 'chosen'])