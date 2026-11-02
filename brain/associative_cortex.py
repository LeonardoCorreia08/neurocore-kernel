'''Modulo: associative_cortex.py'''
import torch
import torch.nn as nn
from core.config_loader import get_config_value

class AssociativeCortex(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int, config: dict | None = None):
        super().__init__()
        total_input_dim = sum(input_dims)
        self.hidden_dim = hidden_dim
        self.membrane_leak = get_config_value(config, "brain.associative_cortex.membrane_leak", 0.8)
        self.spike_threshold = get_config_value(config, "brain.associative_cortex.spike_threshold", 0.5)
        
        # Matriz de projeção multimodal para o espaço associativo
        projection_init_scale = get_config_value(config, "brain.associative_cortex.projection_init_scale", 0.1)
        self.projection_weights = torch.randn(hidden_dim, total_input_dim) * projection_init_scale
        self.membrane_potential = torch.zeros(hidden_dim)

    def forward(self, multimodal_inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Integra múltiplos fluxos de entrada (ex: sensorial + memória) 
        em uma representação cortical unificada.
        """
        concatenated_input = torch.cat(multimodal_inputs, dim=-1)
        
        # Propagação linear das entradas combinadas
        current = torch.matmul(self.projection_weights, concatenated_input)
        
        # Dinâmica simples de integração de membrana com vazamento
        self.membrane_potential = self.membrane_potential * self.membrane_leak + current
        
        # Disparo de spikes associativos (limiar em 0.5)
        spikes = (self.membrane_potential >= self.spike_threshold).float()
        
        # Reset do potencial para os neurônios que dispararam
        self.membrane_potential = torch.where(spikes > 0, torch.zeros_like(self.membrane_potential), self.membrane_potential)
        
        return spikes
