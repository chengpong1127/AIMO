from aimo import AIMOPPOTrainer
from datasets import load_dataset

def main():
    dataset = load_dataset("gsm8k", "main")
    dataset = dataset.rename_columns({"question": "query"})
    dataset = dataset.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.select(range(200))
    test_dataset = test_dataset.select(range(20))

    trainer = AIMOPPOTrainer(
        "/home/wrt/chengpong/AIMO/configs/default_config.yaml",
        train_dataset,
        test_dataset
    )
    trainer.train()

if __name__ == "__main__":
    main()


