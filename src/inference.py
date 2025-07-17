import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

def load_model_for_chat(base_model: str, adapter_path: str):
    # Tokenizer & chat template for inference
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = (
        "{% for msg in messages %}"
        "<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n"
        "{% if loop.last and add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        "{% endfor %}"
    )

    # Load base model then LoRA adapters
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model

def chat(
    tokenizer, model, system_msg: str, user_msg: str,
    temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 200,
):
    # Build prompt
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        add_generation_prompt=True, tokenize=False,
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    out = generator(
        prompt, do_sample=True, temperature=temperature,
        top_p=top_p, max_new_tokens=max_new_tokens, truncation=True,
    )
    return out[0]["generated_text"]
