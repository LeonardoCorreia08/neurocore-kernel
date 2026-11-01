'''Modulo: motor_engine.py'''
import torch
import torch.nn as nn

class MotorEngine(nn.Module):
    def __init__(self):
        super().__init__()

    def decode_action(self, decision_spikes: torch.Tensor) -> int:
        """
        Converte os disparos neurais de decisao em um ID de acao discreto 
        para interagir diretamente com o ambiente virtual (ex: 0: Cima, 1: Baixo, 2: Esquerda, 3: Direita).
        """
        # Identificar quais neuronios motores dispararam
        active_neurons = torch.nonzero(decision_spikes > 0, as_tuple=True)[0]
        
        if len(active_neurons) > 0:
            # Retorna o indice da primeira acao vencedora
            return int(active_neurons[0].item())
        
        # Caso nenhum neuronio dispare no ciclo atual, retorna uma acao neutra ou padrao (0)
        return 0