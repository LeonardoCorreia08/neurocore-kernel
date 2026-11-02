'''Modulo: attention_engine.py'''
import torch
from core.config_loader import get_config_value

class AttentionEngine:
    def __init__(self, dim: int, config: dict | None = None):
        self.dim = dim
        self.salience_threshold = get_config_value(config, "brain.attention.salience_threshold", 0.3)
        self.high_salience_gain = get_config_value(config, "brain.attention.high_salience_gain", 1.8)
        self.low_salience_gain = get_config_value(config, "brain.attention.low_salience_gain", 0.4)
        mask_init_value = get_config_value(config, "brain.attention.mask_init_value", 1.0)
        self.salience_mask = torch.full((dim,), mask_init_value)

    def apply_attention(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """
        Filtra o sinal sensorial: amplifica estimulos de alta relevancia 
        (como perigos ou metas proximas) e atenua ruidos irrelevantes.
        """
        gated_input = torch.where(
            sensory_input > self.salience_threshold,
            sensory_input * self.high_salience_gain,
            sensory_input * self.low_salience_gain,
        )
        return gated_input * self.salience_mask
