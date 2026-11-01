'''Modulo: associative_cortex.py'''
import torch
import torch.nn as nn

class AssociativeCortex(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int):
        super().__init__()
        total_input_dim = sum(input_dims)
        self.hidden_dim = hidden_dim
        
        # Matriz de projeção multimodal para o espaço associativo
        self.projection_weights = torch.randn(hidden_dim, total_input_dim) * 0.1
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
        self.membrane_potential = self.membrane_potential * 0.8 + current
        
        # Disparo de spikes associativos (limiar em 0.5)
        spikes = (self.membrane_potential >= 0.5).float()
        
        # Reset do potencial para os neurônios que dispararam
        self.membrane_potential = torch.where(spikes > 0, torch.zeros_like(self.membrane_potential), self.membrane_potential)
        
        return spikes