import torch
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
from .data_utils import format_dpo_example

logger = logging.getLogger(__name__)

def train(config: dict):
    """
    Initializes and runs the DPO training process based on the provided config.
    """
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_config"]["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = (
        "{% for msg in messages %}"
        "<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n"
        "{% if loop.last and add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        "{% endfor %}"
    )
    
    # 2. Load and process dataset
    raw_dataset = load_dataset(config["model_config"]["dataset_name"], split="train")
    processed_dataset = raw_dataset.map(
        lambda ex: format_dpo_example(ex, tokenizer),
        remove_columns=raw_dataset.column_names,
    )

    # 3. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_config"]["base_model"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False, # Important for training
    )
    model = prepare_model_for_kbit_training(model)
    
    # 4. Setup LoRA
    lora_config = LoraConfig(**config["peft_config"])
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # 5. Setup DPO Trainer
    output_dir = config["model_config"]["output_dir"]
    dpo_config = DPOConfig(
        output_dir=output_dir,
        padding_value=tokenizer.pad_token_id,
        label_pad_token_id=tokenizer.pad_token_id,
        truncation_mode="keep_end",
        **config["training_args"],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        ref_model=None, # TRL will handle creating the reference model
    )

    # 6. Start Training
    logger.info("🚀 Starting DPO training...")
    trainer.train()
    logger.info("✅ Training complete.")
    
    # 7. Save Model and Tokenizer
    logger.info(f"💾 Saving adapters to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("All done!")
