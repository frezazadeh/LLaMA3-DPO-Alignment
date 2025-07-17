# LLaMA-3 DPO Alignment 🚀

This repository provides a straightforward, organized implementation for fine-tuning LLaMA-3 models using **Direct Preference Optimization (DPO)**. The code is split into logical modules for training and inference while keeping all configuration within a single, easy-to-use command-line interface.

### ✨ Key Features

-   **Direct Preference Optimization (DPO):** Uses the `trl` library for efficient preference-based fine-tuning.
-   **Parameter-Efficient Fine-Tuning (PEFT):** Leverages LoRA to train models efficiently with significantly reduced memory and computational costs.
-   **8-bit Quantization:** Integrates `bitsandbytes` to make training large models feasible on consumer-grade GPUs.
-   **Simple & Organized:** The code is split into modules for clarity without adding complex configuration files.
-   **Unified Command-Line Interface:** A single `main.py` script handles both training and inference modes.

---

### 📂 Project Structure

The project is organized to separate concerns while maintaining simplicity.

```
LLaMA3-DPO-Alignment/
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Holds the data formatting function
│   ├── training.py         # Holds the main training logic
│   └── inference.py        # Holds the model loading and chat functions
├── main.py                   # The main script to run training or chat
└── requirements.txt          # Project dependencies
```

---

### 🏁 Getting Started

#### 1. Clone the Repository

```bash
git clone [https://github.com/YourUsername/LLaMA3-DPO-Alignment.git](https://github.com/YourUsername/LLaMA3-DPO-Alignment.git)
cd LLaMA3-DPO-Alignment
```

#### 2. Create a Virtual Environment & Install Dependencies

Using a virtual environment is highly recommended. You will need Python 3.9+ and a version of PyTorch compatible with your CUDA toolkit.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```

---

### 🚀 Usage

All actions are controlled from the `main.py` script using the `--mode` flag.

#### To Train a Model

To start fine-tuning, run the script in `train` mode. It will use the default parameters defined in `main.py` to download the model and dataset and then save the resulting adapters to the `output_dir`.

```bash
python main.py --mode train
```

You can override the default settings by passing command-line arguments:

```bash
python main.py --mode train --base_model "meta-llama/Meta-Llama-3-8B" --output_dir "my_custom_adapters"
```

#### To Chat with Your Model

Once training is complete, run the script in `chat` mode to interact with your fine-tuned model.

```bash
python main.py --mode chat --user_msg "What are the top 3 benefits of DPO for LLM alignment?"
```

You can also provide a custom system prompt:
```bash
python main.py --mode chat \
  --system_msg "You are a machine learning expert who explains complex topics simply." \
  --user_msg "Explain LoRA in two sentences."
```

---

### ⚙️ Configuration

In this setup, all configuration is handled directly through **command-line arguments** in `main.py`. The default values (e.g., learning rate, model names, batch size) are defined in the `argparse` section of the `main.py` script.

To permanently change a default setting, simply edit its `default=` value inside `main.py`.

---

### 📜 License

This project is licensed under the MIT License.
