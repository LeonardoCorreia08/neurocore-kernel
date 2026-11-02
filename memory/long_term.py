'''Modulo: long_term.py'''
import torch
import os
from core.config_loader import get_config_value


class LongTermMemoryManager:
    def __init__(self, storage_dir: str | None = None, config: dict | None = None):
        self.storage_dir = storage_dir if storage_dir is not None else get_config_value(config, "memory.storage_dir", "memory/storage")
        self.map_location = get_config_value(config, "memory.map_location", None)
        self.weights_only = get_config_value(config, "memory.weights_only", None)
        os.makedirs(self.storage_dir, exist_ok=True)

    def store_memory_snapshot(self, memory_name: str, state_dict: dict):
        """Consolida e salva um snapshot estrutural de longo prazo no disco."""
        filepath = os.path.join(self.storage_dir, f"{memory_name}.pt")
        torch.save(state_dict, filepath)
        print(f"[LTM] Memoria de longo prazo consolidada e salva em: {filepath}")

    def load_memory_snapshot(self, memory_name: str) -> dict:
        """Carrega snapshot novo ou legado com localização configurável."""
        filepath = os.path.join(self.storage_dir, f"{memory_name}.pt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot de memoria '{memory_name}' nao encontrado.")
        kwargs = {}
        if self.map_location is not None:
            kwargs['map_location'] = self.map_location
        if self.weights_only is not None:
            kwargs['weights_only'] = self.weights_only
        try:
            state_dict = torch.load(filepath, **kwargs)
        except TypeError:
            kwargs.pop('weights_only', None)
            state_dict = torch.load(filepath, **kwargs)
        print(f"[LTM] Memoria de longo prazo carregada de: {filepath}")
        return state_dict
