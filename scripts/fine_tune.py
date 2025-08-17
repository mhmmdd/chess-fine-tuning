import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M on the ChessInstruct dataset using Unsloth.")
    parser.add_argument("--dataset_path", type=str, default="data/chess_instruct_chatml.json", help="Path to the prepared dataset.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m-it", help="The model to fine-tune.")
    parser.add_argument("--output_dir", type=str, default="models/gemma-3-270m-chess", help="Directory to save the fine-tuned model.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    args = parser.parse_args()

    # 1) Dataset yükle
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # 2) Model + tokenizer (Unsloth)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 3) conversations -> text (tek bir string) dönüşümü
    #    (Batched map ile düz string listesi üretip sadece "text" kolonunu bırakıyoruz)
    orig_cols = dataset.column_names
    def to_text(batch):
        # batch["conversations"] = list[ list[{"role":..., "content":...}, ...] ]
        texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in batch["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True, remove_columns=orig_cols)

    # 4) LoRA ayarı
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

    # 5) Trainer (artık dataset_text_field="text" kullanıyoruz; formatting_func YOK)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=1,   # WSL/Windows'ta çoklu süreç bazen sorun çıkarıyor; istersen sonra artır
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
            optim="adamw_8bit",     # bitsandbytes yoksa "adamw_torch_fused" deneyin
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
    model.save_pretrained(args.output_dir)   # LoRA adapter kaydı
    tokenizer.save_pretrained(args.output_dir)

    # İsteğe bağlı: LoRA'yı birleştirip tek parça fp16 model de çıkarabilirsiniz:
    # model.save_pretrained_merged(args.output_dir + "-merged", tokenizer, save_method="merged_16bit")

    print("Model saved successfully.")

if __name__ == "__main__":
    main()
