from aimoppo import AIMOPPOTrainer
from datasets import load_dataset, Dataset
import os

def main():
    os.system("rm -rf models/checkpoints/*")
    dataset = load_dataset("gsm8k", "main")
    dataset = dataset.rename_columns({"question": "query"})
    dataset = dataset.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
    train_dataset = dataset["train"]
    val_dataset = Dataset.from_csv("data/val.csv")
    val_dataset = val_dataset.rename_column("problem", "query")
    val_dataset = val_dataset.remove_columns(["id"])

    train_dataset = train_dataset.map(lambda x: {"answer": x["answer"].replace(",", "")})

    train_dataset = train_dataset.shuffle()
    val_dataset = val_dataset.shuffle()


    trainer = AIMOPPOTrainer(
        "configs/costar_cot_1shot.yaml",
        train_dataset,
        val_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()


# accelerate launch aimoppoTrainingMultiGPU.py