'''Modulo: motor_engine.py'''
import torch
import torch.nn as nn
from core.config_loader import get_config_value

class MotorEngine(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.selection_policy = get_config_value(config, "brain.motor.selection_policy", "first_active")
        self.no_spike_action = get_config_value(config, "brain.motor.no_spike_action", 0)
        self.action_ids = get_config_value(config, "brain.motor.action_ids", None)

    def decode_action(self, decision_spikes: torch.Tensor) -> int:
        """
        Converte os disparos neurais de decisao em um ID de acao discreto 
        para interagir diretamente com o ambiente virtual (ex: 0: Cima, 1: Baixo, 2: Esquerda, 3: Direita).
        """
        # Identificar quais neuronios motores dispararam
        active_neurons = torch.nonzero(decision_spikes > 0, as_tuple=True)[0]
        
        if len(active_neurons) > 0:
            if self.selection_policy == "max_active":
                neuron_index = int(torch.argmax(decision_spikes).item())
            else:
                # first_active (inclui políticas desconhecidas) mantém semântica legada.
                neuron_index = int(active_neurons[0].item())
            if self.action_ids is not None and neuron_index < len(self.action_ids):
                return int(self.action_ids[neuron_index])
            return neuron_index
        
        # Caso nenhum neuronio dispare no ciclo atual, retorna uma acao neutra ou padrao (0)
        return int(self.no_spike_action)
