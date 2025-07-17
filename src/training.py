import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from src.data_processing import format_example # Import from our new file

logger = logging.getLogger(__name__)

def train_model(
    base_model: str,
    dataset_name: str,
    output_dir: str,
    peft_r: int = 16,
    peft_alpha: int = 16,
    peft_dropout: float = 0.05,
    batch_size: int = 1,
    grad_accum: int = 32,
    learning_rate: float = 5e-5,
    max_steps: int = 40,
    warmup_steps: int = 10,
    bf16: bool = True,
):
    # Tokenizer & chat template
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = (
        "{% for msg in messages %}"
        "<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n"
        "{% if loop.last and add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        "{% endfor %}"
    )

    # Load and preprocess dataset
    raw = load_dataset(dataset_name, split="train")
    processed = raw.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=raw.column_names,
        batched=False,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map={'': 'cpu'} if not torch.cuda.is_available() else 'auto',
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,)
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=peft_r, lora_alpha=peft_alpha, lora_dropout=peft_dropout,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # DPO training configuration
    dpo_cfg = DPOConfig(
        output_dir=output_dir, per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, gradient_checkpointing=True,
        learning_rate=learning_rate, lr_scheduler_type="cosine", max_steps=max_steps,
        logging_steps=1, optim="adamw_torch", warmup_steps=warmup_steps,
        padding_value=tokenizer.pad_token_id, label_pad_token_id=tokenizer.pad_token_id,
        truncation_mode="keep_end", bf16=True, fp16=False,
    )

    trainer = DPOTrainer(
        model=model, args=dpo_cfg, train_dataset=processed,
        tokenizer=tokenizer, peft_config=lora_cfg, ref_model=None,
    )

    logger.info("Starting DPO training...")
    trainer.train()
    logger.info("Training complete. Saving adapters and tokenizer...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
