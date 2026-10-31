'''Modulo: short_term.py'''
import torch

class ShortTermMemory:
    def __init__(self, hidden_dim: int, history_steps: int = 5):
        self.hidden_dim = hidden_dim
        self.history_steps = history_steps
        # Buffer circular para armazenar os ultimos spikes da camada oculta
        self.buffer = torch.zeros(history_steps, hidden_dim)
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