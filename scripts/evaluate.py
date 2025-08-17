from unsloth import FastLanguageModel
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Gemma model.")
    parser.add_argument("--model_path", type=str, default="models/gemma-3-270m-chess", help="Path to the fine-tuned model directory.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    args = parser.parse_args()

    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Example prompt based on the ChessInstruct dataset format
    prompt = [
        {"role": "system", "content": "You are a helpful chess assistant."},
        {"role": "user", "content": "What is the best move for white in the position: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3?"}
    ]

    # Format the prompt
    inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # Generate the response
    print("Generating response...")
    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True)
    response = tokenizer.batch_decode(outputs)
    
    print("\n--- Prompt ---")
    print(prompt[1]['content'])
    print("\n--- Model Response ---")
    # The response includes the prompt, so we extract just the generated part
    print(response[0].split('[/INST]')[-1].strip())
    print("\n")

if __name__ == "__main__":
    main()