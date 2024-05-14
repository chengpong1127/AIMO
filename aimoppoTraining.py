

def training_func():
    from aimoppo import AIMOPPOTrainer, TrainerConfig
    
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main")
    dataset = dataset.rename_columns({"question": "query"})
    dataset = dataset.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.select(range(200))
    test_dataset = test_dataset.select(range(20))
    

    
    config = TrainerConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_path="/home/wrt/chengpong/AIMO/prompts/simpleCOT.txt",
        save_dir="/home/wrt/chengpong/AIMO/models",
        wandb_project_name="AIMO",
        
        #python: lvwerra/python-interpreter
        text_env_tools = [],
        epochs=10,
        train_batch_size=4,
        train_mini_batch_size=1,
        val_batch_size=4,
        gradient_accumulation_steps=4,
    )
    trainer = AIMOPPOTrainer(
        config,
        train_dataset,
        test_dataset
    )
    trainer.train()

if __name__ == "__main__":
    training_func()


