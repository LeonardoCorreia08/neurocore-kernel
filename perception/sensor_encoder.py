'''Modulo: sensor_encoder.py'''
import torch

class SensorEncoder:
    def __init__(self, max_rate: float = 100.0):
        self.max_rate = max_rate

    def encode_rate(self, continuous_values: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Converte valores contínuos (0.0 a 1.0) em probabilidade de disparo (spikes) 
        proporcional à taxa de disparos (rate coding).
        """
        # Normalizar valores entre 0 e 1
        normalized = torch.clamp(continuous_values, 0.0, 1.0)
        # Gerar probabilidade de pulso baseada no valor
        probabilities = normalized * self.max_rate * dt
        spikes = torch.rand_like(probabilities) < probabilities
        return spikes.float()