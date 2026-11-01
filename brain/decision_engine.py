'''Modulo: decision_engine.py'''
import torch
import torch.nn as nn

class DecisionEngine(nn.Module):
    def __init__(self, num_choices: int, inhibition_strength: float = 0.5):
        super().__init__()
        self.num_choices = num_choices
        self.inhibition_strength = inhibition_strength
        self.membrane_potential = torch.zeros(num_choices)

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        Executa a seleção competitiva de ações através de inibição lateral 
        entre as populações neurais de decisão.
        """
        # Acumular potencial com base nas entradas recebidas
        self.membrane_potential = self.membrane_potential * 0.7 + input_spikes
        
        # Aplicar inibição lateral (o neurônio mais ativo suprime os concorrentes)
        max_val, _ = torch.max(self.membrane_potential, dim=-1, keepdim=True)
        self.membrane_potential = torch.where(
            self.membrane_potential < max_val, 
            self.membrane_potential - self.inhibition_strength, 
            self.membrane_potential
        )
        
        # Gerar disparos de decisão quando o potencial atinge o limiar
        decisions = (self.membrane_potential >= 1.0).float()
        
        # Reset do potencial para os neurônios que dispararam
        self.membrane_potential = torch.where(
            decisions > 0, 
            torch.zeros_like(self.membrane_potential), 
            self.membrane_potential
        )
        
        return decisions