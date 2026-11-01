'''Modulo: attention_engine.py'''
import torch

class AttentionEngine:
    def __init__(self, dim: int):
        self.dim = dim
        self.salience_mask = torch.ones(dim)

    def apply_attention(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """
        Filtra o sinal sensorial: amplifica estimulos de alta relevancia 
        (como perigos ou metas proximas) e atenua ruidos irrelevantes.
        """
        gated_input = torch.where(sensory_input > 0.3, sensory_input * 1.8, sensory_input * 0.4)
        return gated_input * self.salience_mask