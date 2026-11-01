'''Modulo: long_term.py'''
import torch
import os

class LongTermMemoryManager:
    def __init__(self, storage_dir: str = "memory/storage"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def store_memory_snapshot(self, memory_name: str, state_dict: dict):
        """Consolida e salva um snapshot estrutural de longo prazo no disco."""
        filepath = os.path.join(self.storage_dir, f"{memory_name}.pt")
        torch.save(state_dict, filepath)
        print(f"[LTM] Memoria de longo prazo consolidada e salva em: {filepath}")

    def load_memory_snapshot(self, memory_name: str) -> dict:
        """Carrega uma memoria de longo prazo consolidada anteriormente."""
        filepath = os.path.join(self.storage_dir, f"{memory_name}.pt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot de memoria '{memory_name}' nao encontrado.")
        state_dict = torch.load(filepath)
        print(f"[LTM] Memoria de longo prazo carregada de: {filepath}")
        return state_dict