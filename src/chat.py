import yaml
import argparse
from inference.chatbot import load_model_for_chat, generate_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a DPO-tuned model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to the configuration YAML file to load model paths.",
    )
    parser.add_argument(
        "--user_msg",
        type=str,
        required=True,
        help="Your message to the chatbot.",
    )
    parser.add_argument(
        "--system_msg",
        type=str,
        default="You are a helpful assistant.",
        help="The system prompt.",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config["model_config"]
    
    print("Loading model... Please wait.")
    model, tokenizer = load_model_for_chat(
        base_model_path=model_config["base_model"],
        adapter_path=model_config["output_dir"],
    )
    
    print("Model loaded. Generating response...\n")
    response = generate_response(
        model,
        tokenizer,
        args.system_msg,
        args.user_msg,
        config
    )
    
    # Cleanly print just the assistant's response
    assistant_response = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    
    print("---")
    print(f"👤 You: {args.user_msg}")
    print(f"🤖 Assistant: {assistant_response}")
    print("---\n")
