from datasets import load_dataset

def format_example(example, tokenizer):
    # Construct a single-example mapping for DPO training
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages.append({"role": "user", "content": example["question"]})

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    chosen = example["chosen"] + tokenizer.eos_token
    rejected = example["rejected"] + tokenizer.eos_token

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
