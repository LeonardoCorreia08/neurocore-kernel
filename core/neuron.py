'''Modulo: core/neuron.py'''
import torch
import torch.nn as nn

from core.config_loader import get_config_value

class LeakyIntegrateAndFire(nn.Module):
    def __init__(self, num_neurons: int, threshold: float = 1.0, decay: float = 0.9, config: dict | None = None):
        super().__init__()
        self.num_neurons = num_neurons
        self.threshold = get_config_value(config, "neuron.threshold", threshold)
        self.decay = get_config_value(config, "neuron.decay", decay)
        self.rest_potential = get_config_value(config, "neuron.rest_potential", 0.0)
        self.mem = None

    def _init_state_if_needed(self, reference_tensor: torch.Tensor):
        """Garante compatibilidade de lote (batch) e hardware (CPU/GPU)."""
        if self.mem is None or self.mem.shape != reference_tensor.shape or self.mem.device != reference_tensor.device:
            self.mem = torch.full_like(
                reference_tensor, float(self.rest_potential), device=reference_tensor.device
            )

    def integrate(self, input_current: torch.Tensor):
        """Etapa 1: Integração temporal da membrana com fator de vazamento (Leak)."""
        self.mem = self.rest_potential + (self.mem - self.rest_potential) * self.decay + input_current

    def fire(self) -> torch.Tensor:
        """Etapa 2: Aplicação do limiar e geração do pulso (Spike)."""
        return (self.mem >= self.threshold).float()

    def apply_reset(self, spikes: torch.Tensor):
        """Etapa 3: Reset local (zera a voltagem apenas dos neurônios que dispararam)."""
        reset_values = torch.full_like(self.mem, float(self.rest_potential))
        self.mem = torch.where(spikes > 0, reset_values, self.mem)

    def forward(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ciclo de vida completo do neurônio em um passo de tempo."""
        self._init_state_if_needed(input_current)
        
        self.integrate(input_current)
        spikes = self.fire()
        self.apply_reset(spikes)
        
        return spikes, self.mem

    def reset_state(self):
        """Reset global (usado para limpar o ambiente entre episódios)."""
        if self.mem is not None:
            self.mem.fill_(float(self.rest_potential))
