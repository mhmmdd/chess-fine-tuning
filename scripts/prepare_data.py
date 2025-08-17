import argparse
import os
from datasets import load_dataset
# Unsloth'un kurulu olduğunu varsayıyoruz, requirements.txt'e eklenecek.
# from unsloth.chat_templates import standardize_data_formats

def convert_to_chatml(example):
    """
    Converts a single example from the ChessInstruct dataset to ChatML format.
    """
    return {
        "conversations": [
            {"role": "system", "content": example.get("task", "")},
            {"role": "user", "content": example.get("input", "")},
            {"role": "assistant", "content": example.get("expected_output", "")}
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare the Thytu/ChessInstruct dataset in ChatML format.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the output JSON file.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "chess_instruct_chatml.json")

    print("Loading Thytu/ChessInstruct dataset...")
    try:
        dataset = load_dataset("Thytu/ChessInstruct", split="train")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Resimdeki standardize_data_formats fonksiyonu unsloth'a özel.
    # Eğer unsloth'un kendi eğitim betiği kullanılmayacaksa bu adım atlanabilir
    # veya manuel olarak yapılabilir. Şimdilik temel formatlamayı yapıyoruz.
    # dataset = standardize_data_formats(dataset)

    print("Converting dataset to ChatML format...")
    # num_proc > 1 ile hızlandırılabilir
    dataset = dataset.map(convert_to_chatml, num_proc=4)

    print(f"Saving processed dataset to {output_file}...")
    dataset.to_json(output_file, orient="records", lines=True)

    print("Data preparation complete.")
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()