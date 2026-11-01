'''Modulo: perception/sensor_encoder.py'''
import torch
import numpy as np

class SensorEncoder:
    def __init__(self, max_rate: float = 100.0, dt: float = 0.01):
        """
        Codificador Sensorial: Converte estados analógicos do mundo real 
        em sinais (Corrente ou Spikes) para o cérebro (SNN).
        """
        self.max_rate = max_rate
        self.dt = dt

    def _to_tensor(self, data) -> torch.Tensor:
        """Garante que a entrada do ambiente (numpy) seja convertida com segurança."""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=torch.float32)
        return data.float()

    def encode_rate(self, continuous_values) -> torch.Tensor:
        """
        1. Rate Coding (Poisson): Converte valores em probabilidade de disparo (spikes) 
        proporcional à taxa de disparos.
        """
        tensor_values = self._to_tensor(continuous_values)
        
        # Como o ambiente pode enviar valores negativos (ex: direção esquerda/baixo),
        # usamos a magnitude (abs) para calcular a força do disparo (probabilidade).
        magnitude = torch.abs(tensor_values)
        
        # Normalizar a magnitude entre 0.0 e 1.0
        normalized = torch.clamp(magnitude, 0.0, 1.0)
        
        # Gerar probabilidade de pulso baseada no valor
        probabilities = normalized * self.max_rate * self.dt
        spikes = (torch.rand_like(probabilities) < probabilities).float()
        
        # Restaura o sinal (positivo ou negativo) para que a rede saiba a direção.
        # Exemplo: Spike +1.0 (frente) ou Spike -1.0 (trás)
        spikes = spikes * torch.sign(tensor_values)
        
        return spikes

    def encode_current(self, continuous_values) -> torch.Tensor:
        """
        2. Injeção de Corrente (Current Coding): Envia o sinal analógico 
        diretamente como corrente para a camada Sensorial LIF integrar.
        Geralmente mais estável e rápido para treinar agentes 2D em Reinforcement Learning.
        """
        return self._to_tensor(continuous_values)