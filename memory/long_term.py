"""Utilitário de snapshot seguro para artefatos tensor/primitivo de longo prazo."""

import os
import tempfile
from pathlib import Path

import torch


class LongTermMemoryManager:
    def __init__(self, storage_dir: str = "memory/storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_path(self, memory_name: str) -> Path:
        name = Path(memory_name)
        if name.is_absolute() or ".." in name.parts or len(name.parts) != 1:
            raise ValueError("memory_name deve ser um nome simples sem diretórios")
        if name.suffix != ".pt":
            name = name.with_suffix(".pt")
        return self.storage_dir / name

    def store_memory_snapshot(self, memory_name: str, state_dict: dict):
        """Write a checkpoint atomically so interruption cannot leave a partial file."""
        filepath = self._snapshot_path(memory_name)
        fd, temp_name = tempfile.mkstemp(prefix=f".{filepath.name}.", suffix=".tmp", dir=filepath.parent)
        os.close(fd)
        try:
            torch.save(state_dict, temp_name)
            os.replace(temp_name, filepath)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return str(filepath)

    def load_memory_snapshot(self, memory_name: str, map_location: str | torch.device = "cpu") -> dict:
        """Load only tensors and primitive values; never unpickle arbitrary classes."""
        filepath = self._snapshot_path(memory_name)
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot de memoria '{memory_name}' não encontrado.")
        state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
        if not isinstance(state_dict, dict):
            raise ValueError("O snapshot deve conter um dicionário de tensores/valores simples")
        return state_dict
