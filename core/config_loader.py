'''Modulo: config_loader.py'''
import yaml
import os

def load_brain_config(config_path: str = "config/brain_config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado em: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config