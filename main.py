import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Keep this at the top
import logging
import argparse


from src.training import train_model
from src.inference import load_model_for_chat, chat

def setup_logger(level=logging.INFO):
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s] %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = setup_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or chat with a DPO-fine-tuned LLaMA-3 model"
    )
    parser.add_argument(
        "--mode", choices=["train", "chat"], required=True,
        help="Choose 'train' to fine-tune or 'chat' to run inference"
    )
    parser.add_argument("--base_model", default="unsloth/llama-3-8B")
    parser.add_argument("--dataset", default="Intel/orca_dpo_pairs")
    parser.add_argument("--output_dir", default="llama3_dpo")
    parser.add_argument("--adapter_path", default="llama3_dpo")
    parser.add_argument(
        "--system_msg", default="You are a helpful assistant.",
        help="System prompt for chat mode"
    )
    parser.add_argument("--user_msg", help="User query for chat mode")
    args = parser.parse_args()

    if args.mode == "train":
        # Call the imported train_model function
        train_model(
            base_model=args.base_model,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
        )
    else:
        # Call the imported chat functions
        tokenizer, model = load_model_for_chat(
            args.base_model, args.adapter_path
        )
        response = chat(
            tokenizer, model,
            args.system_msg, args.user_msg
        )
        print(response)
