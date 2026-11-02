'''Modulo: memory/manager.py'''
import os
import torch
from core.config_loader import get_config_value
from core.spike_engine import SpikeEngine


class MemoryManager:
    def __init__(self, engine: SpikeEngine, save_dir: str | None = None, config: dict | None = None):
        """Gerenciador de memória cognitiva, com configuração opcional."""
        self.engine = engine
        self.config = config
        self.save_dir = save_dir if save_dir is not None else get_config_value(config, "memory.checkpoint_dir", "checkpoints/")
        self.checkpoint_filename = get_config_value(config, "memory.checkpoint_filename", "latest_brain_engram.pt")
        self.map_location = get_config_value(config, "memory.map_location", None)
        self.weights_only = get_config_value(config, "memory.weights_only", None)
        os.makedirs(self.save_dir, exist_ok=True)
        self.episodic_buffer = []
        self.max_buffer_size = get_config_value(config, "memory.episodic_buffer_max_size", 1000)

    def store_experience(self, sensory_state: list[float], action: int, reward: float):
        experience = {"state": sensory_state, "action": action, "reward": reward}
        self.episodic_buffer.append(experience)
        if len(self.episodic_buffer) > self.max_buffer_size:
            self.episodic_buffer.pop(0)

    def clear_short_term_memory(self):
        self.episodic_buffer.clear()

    def consolidate_memory(self, memory_name: str | None = None):
        """Salva as matrizes esparsas atuais no formato legado layer_N."""
        memory_name = self.checkpoint_filename if memory_name is None else memory_name
        filepath = os.path.join(self.save_dir, memory_name)
        engram_data = {}
        for idx, synapse_layer in enumerate(self.engine.synapses):
            engram_data[f"layer_{idx}"] = {
                'indices': synapse_layer.sparse_weights.indices().cpu(),
                'values': synapse_layer.sparse_weights.values().cpu(),
                'size': synapse_layer.sparse_weights.size()
            }
        # Metadados adicionais; chaves layer_N permanecem idênticas para leitores antigos.
        engram_data['_metadata'] = {'format': 'neurocore-memory', 'version': 1}
        torch.save(engram_data, filepath)
        print(f"[Memória] Engrama consolidado com sucesso em: {filepath}")

    def _load_checkpoint(self, filepath):
        kwargs = {}
        if self.map_location is not None:
            kwargs['map_location'] = self.map_location
        if self.weights_only is not None:
            kwargs['weights_only'] = self.weights_only
        try:
            return torch.load(filepath, **kwargs)
        except TypeError:
            # Compatibilidade com versões PyTorch sem suporte a weights_only.
            kwargs.pop('weights_only', None)
            return torch.load(filepath, **kwargs)

    def retrieve_memory(self, memory_name: str | None = None):
        """Restaura engramas novos ou snapshots legados sem metadados."""
        memory_name = self.checkpoint_filename if memory_name is None else memory_name
        filepath = os.path.join(self.save_dir, memory_name)
        if not os.path.exists(filepath):
            print(f"[Memória] Engrama {memory_name} não encontrado. Iniciando com cérebro em branco.")
            return False
        engram_data = self._load_checkpoint(filepath)
        restored_layers = []
        if not isinstance(engram_data, dict):
            raise ValueError("Formato de checkpoint de memória inválido.")
        for idx, synapse_layer in enumerate(self.engine.synapses):
            key = f"layer_{idx}"
            if key not in engram_data:
                raise ValueError(f"Checkpoint incompleto: ausente {key}.")
            layer_data = engram_data[key]
            if tuple(layer_data["size"]) != tuple(synapse_layer.sparse_weights.size()):
                raise ValueError(f"Dimensão incompatível no checkpoint da camada {idx}.")
            target = synapse_layer.sparse_weights
            restored_layers.append(torch.sparse_coo_tensor(
                layer_data["indices"].to(device=target.device),
                layer_data["values"].to(device=target.device, dtype=target.dtype),
                size=layer_data["size"],
                device=target.device,
            ).coalesce())
        if len([key for key in engram_data if key.startswith("layer_")]) != len(self.engine.synapses):
            raise ValueError("Número de camadas do checkpoint não corresponde ao engine.")
        for synapse_layer, restored_weights in zip(self.engine.synapses, restored_layers):
            synapse_layer.sparse_weights = restored_weights
        print(f"[Memória] Engrama {memory_name} recuperado e associado à rede com sucesso!")
        return True
