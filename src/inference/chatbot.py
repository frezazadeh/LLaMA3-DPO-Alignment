import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

def load_model_for_chat(base_model_path: str, adapter_path: str):
    """
    Loads the base model and applies the fine-tuned PEFT adapters.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.chat_template = (
        "{% for msg in messages %}"
        "<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n"
        "{% if loop.last and add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        "{% endfor %}"
    )

    return model, tokenizer

def generate_response(
    model,
    tokenizer,
    system_msg: str,
    user_msg: str,
    config: dict,
):
    """
    Generates a response from the model given system and user messages.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    output = generator(
        prompt,
        do_sample=True,
        truncation=True,
        **config["inference_config"],
    )
    
    return output[0]["generated_text"]
