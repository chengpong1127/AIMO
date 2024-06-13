from aimoppo import AIMOPPOTrainer
from datasets import load_dataset, Dataset
import os
import re

def main():
    os.system("rm -rf models/checkpoints/*")
    dataset = load_dataset("json", data_dir="data/MATH")
    def is_real_number(text):
        try:
            float(text)
            return True
        except Exception:
            return False
    def extract_answer(text):
        try:
            match = re.search(r"\\boxed{(.+?)}", text)
            return match.group(1)
        except Exception:
            return None

    dataset_with_answer = dataset.map(lambda x: {"problem": x["problem"], "answer": extract_answer(x["solution"])})
    dataset_with_answer = dataset_with_answer.filter(lambda x: is_real_number(x["answer"]))
    dataset_with_answer = dataset_with_answer.rename_column("problem", "query")


    train_dataset = dataset_with_answer['train']
    val_dataset = dataset_with_answer['test']

    test_dataset = Dataset.from_csv("data/val.csv")
    test_dataset = test_dataset.rename_column("problem", "query")
    test_dataset = test_dataset.remove_columns(["id"])

    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    train_dataset = train_dataset.select(range(1000))
    val_dataset = val_dataset.select(range(20))


    trainer = AIMOPPOTrainer(
        "configs/costar_cot_1shot.yaml",
        train_dataset,
        val_dataset,
        test_dataset,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()


# accelerate launch aimoppoTrainingMultiGPU.py