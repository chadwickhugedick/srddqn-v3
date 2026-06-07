import yaml
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        # try to find it relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.yaml")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
