'''Modulo: short_term.py'''
import torch
from core.config_loader import get_config_value


class ShortTermMemory:
    def __init__(self, hidden_dim: int | None = None, history_steps: int | None = None, config: dict | None = None):
        configured_dim = get_config_value(config, "memory.short_term_hidden_dim", None)
        configured_steps = get_config_value(config, "memory.history_steps", 5)
        self.hidden_dim = hidden_dim if hidden_dim is not None else configured_dim
        if self.hidden_dim is None:
            raise TypeError("hidden_dim é obrigatório quando memory.short_term_hidden_dim não está configurado")
        self.history_steps = history_steps if history_steps is not None else configured_steps
        # Buffer circular para armazenar os ultimos spikes da camada oculta
        self.buffer = torch.zeros(self.history_steps, self.hidden_dim)
        self.pointer = 0

    def reset(self):
        self.buffer.zero_()
        self.pointer = 0

    def update(self, spk_hidden: torch.Tensor):
        """Armazena o padrao de spikes atual no buffer de curto prazo."""
        self.buffer[self.pointer] = spk_hidden.detach()
        self.pointer = (self.pointer + 1) % self.history_steps

    def get_context(self) -> torch.Tensor:
        """Retorna a media temporal da memoria de curto prazo (contexto)."""
        return torch.mean(self.buffer, dim=0)
