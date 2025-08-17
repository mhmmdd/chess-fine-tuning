import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import argparse

# We define the formatting function that will be used by the SFTTrainer
# This function takes an example and formats it into the prompt style the model expects.
def formatting_prompts_func(example):
    # The 'conversations' field is already a list of dictionaries.
    # We just need to ensure the tokenizer can process it.
    # The tokenizer will apply the chat template.
    return example["conversations"]

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M on the ChessInstruct dataset using Unsloth.")
    parser.add_argument("--dataset_path", type=str, default="data/chess_instruct_chatml.json", help="Path to the prepared dataset.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m-it", help="The model to fine-tune.")
    parser.add_argument("--output_dir", type=str, default="models/gemma-3-270m-chess", help="Directory to save the fine-tuned model.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Load the model and tokenizer using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # None will auto-detect the best dtype
        load_in_4bit=True,
    )

    # Configure the model for LoRA fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        max_seq_length=args.max_seq_length,
    )

    # Set up the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func, # Use formatting_func instead of dataset_text_field
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=args.output_dir,
        ),
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()