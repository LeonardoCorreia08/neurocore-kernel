'''Modulo: decision_engine.py'''
import torch
import torch.nn as nn
from core.config_loader import get_config_value

class DecisionEngine(nn.Module):
    def __init__(self, num_choices: int, inhibition_strength: float = 0.5, config: dict | None = None):
        super().__init__()
        self.num_choices = num_choices
        self.inhibition_strength = get_config_value(config, "brain.decision.inhibition_strength", inhibition_strength)
        self.membrane_leak = get_config_value(config, "brain.decision.membrane_leak", 0.7)
        self.spike_threshold = get_config_value(config, "brain.decision.spike_threshold", 1.0)
        self.membrane_potential = torch.zeros(num_choices)

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        Executa a seleção competitiva de ações através de inibição lateral 
        entre as populações neurais de decisão.
        """
        # Acumular potencial com base nas entradas recebidas
        self.membrane_potential = self.membrane_potential * self.membrane_leak + input_spikes
        
        # Aplicar inibição lateral (o neurônio mais ativo suprime os concorrentes)
        max_val, _ = torch.max(self.membrane_potential, dim=-1, keepdim=True)
        self.membrane_potential = torch.where(
            self.membrane_potential < max_val, 
            self.membrane_potential - self.inhibition_strength, 
            self.membrane_potential
        )
        
        # Gerar disparos de decisão quando o potencial atinge o limiar
        decisions = (self.membrane_potential >= self.spike_threshold).float()
        
        # Reset do potencial para os neurônios que dispararam
        self.membrane_potential = torch.where(
            decisions > 0, 
            torch.zeros_like(self.membrane_potential), 
            self.membrane_potential
        )
        
        return decisions
