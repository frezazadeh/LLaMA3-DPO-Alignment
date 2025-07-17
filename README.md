# LLaMA-3 DPO Alignment 🚀

This repository provides an end-to-end toolkit for fine-tuning LLaMA-3 models using **Direct Preference Optimization (DPO)**. The code is structured for clarity, maintainability, and ease of experimentation, allowing you to align powerful language models to your specific needs and preferences.

### ✨ Key Features

-   **Direct Preference Optimization (DPO):** Uses the `trl` library for efficient DPO training, an effective alternative to reinforcement learning with human feedback (RLHF).
-   **Parameter-Efficient Fine-Tuning (PEFT):** Leverages LoRA to fine-tune models efficiently, significantly reducing memory and computational requirements.
-   **8-bit Quantization:** Integrates `bitsandbytes` for 8-bit model loading, making it possible to train large models on consumer-grade GPUs.
-   **Configuration-Driven:** All hyperparameters and paths are managed via a central `YAML` file, so no code changes are needed to run new experiments.
-   **Modular & Professional Structure:** Code is organized into distinct modules for data processing, training, and inference, following best practices.
-   **Ready-to-Use Scripts:** Includes simple command-line scripts for both training a new model and chatting with your fine-tuned adapter.

---

### 📂 Project Structure

```
LLaMA3-DPO-Alignment/
├── src/
│   ├── training/
│   │   ├── trainer.py       # Main training logic
│   │   └── data_utils.py    # Data formatting utilities
│   ├── inference/
│   │   └── chatbot.py       # Chat and model loading logic
│   ├── train.py             # Executable script for training
│   └── chat.py              # Executable script for inference
├── configs/
│   └── training_config.yaml # All hyperparameters
├── requirements.txt         # Python dependencies
└── README.md                # You are here!
```

---

### 🏁 Getting Started

#### 1. Clone the Repository

```bash
git clone https://github.com/frezazadeh/LLaMA3-DPO-Alignment.git
cd LLaMA3-DPO-Alignment
```

#### 2. Create a Virtual Environment & Install Dependencies

It's highly recommended to use a virtual environment. You will need Python 3.9+ and a version of PyTorch compatible with your CUDA toolkit.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```

---

### ⚙️ Configuration

Before running, you can customize your training session by editing `configs/training_config.yaml`. Here you can set:
-   `base_model`: The base LLaMA-3 model to fine-tune.
-   `dataset_name`: The preference dataset from Hugging Face Hub.
-   `output_dir`: Where to save the trained LoRA adapters.
-   `peft_config`: LoRA parameters like `r` and `alpha`.
-   `training_args`: DPO hyperparameters like learning rate, batch size, and number of steps.

---

### 🚀 Usage

#### To Train a Model

Run the training script. It will automatically read the settings from the YAML config, download the model and dataset, and save the resulting adapters to the specified `output_dir`.

```bash
python src/train.py --config configs/training_config.yaml
```
Training progress will be logged to the console, and you can monitor it using TensorBoard by running `tensorboard --logdir=llama3_dpo_adapters`.

#### To Chat with Your Model

Once training is complete, you can chat with your aligned model using the inference script. It loads the base model and applies your saved LoRA adapters.

```bash
python src/chat.py --user_msg "What are the top 3 benefits of DPO over PPO for LLM alignment?"
```

You can also provide a custom system prompt:
```bash
python src/chat.py \
  --system_msg "You are a machine learning expert who explains complex topics simply." \
  --user_msg "Explain LoRA in two sentences."
```

---

### 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
