'''Modulo: monitoring/experiment_logger.py'''
import os
import sys
import json
import time
import random
import platform
import torch


def _settings(config):
    if not config:
        return {}
    return config.get("monitoring", config).get("experiment_logger", {})


class ExperimentLogger:
    @staticmethod
    def set_seed(seed: int = 42, config: dict = None):
        """Fixa as sementes de aleatoriedade para garantir reprodutibilidade."""
        seed = _settings(config).get("seed", seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[Logger] Semente (Seed) fixada em: {seed}")

    @staticmethod
    def save_experiment_log(experiment_name: str, params: dict, seed: int = 42, output_dir: str = "logs", config: dict = None):
        """Gera e salva um log completo de reprodutibilidade em JSON."""
        settings = _settings(config)
        seed = settings.get("seed", seed)
        output_dir = settings.get("output_dir", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        hardware_info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": os.cpu_count()
        }

        if torch.cuda.is_available():
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)

        log_data = {
            "experiment_name": experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "hardware_and_env": hardware_info,
            "hyperparameters": params
        }

        filename = f"{output_dir}/{experiment_name}_log_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        print(f"[Logger] Log de reprodutibilidade salvo em: '{filename}'")
        return filename
