from aimoppo import AIMOPPOTrainer
from datasets import load_dataset
import os

def main():
    os.system("rm -rf models/checkpoints/*")
    dataset = load_dataset("gsm8k", "main")
    dataset = dataset.rename_columns({"question": "query"})
    dataset = dataset.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    train_dataset = train_dataset.map(lambda x: {"answer": x["answer"].replace(",", "")})
    test_dataset = test_dataset.map(lambda x: {"answer": x["answer"].replace(",", "")})

    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    train_dataset = train_dataset.select(range(200))
    test_dataset = test_dataset.select(range(16))

    trainer = AIMOPPOTrainer(
        "configs/costar_cot_1shot.yaml",
        train_dataset,
        test_dataset
    )
    trainer.train()

if __name__ == "__main__":
    main()


