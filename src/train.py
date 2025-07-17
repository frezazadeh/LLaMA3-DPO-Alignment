import yaml
import logging
import argparse
from training.trainer import train

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using DPO.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()

    setup_logger()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
