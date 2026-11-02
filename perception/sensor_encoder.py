'''Modulo: perception/sensor_encoder.py'''
import torch
import numpy as np

class SensorEncoder:
    def __init__(self, max_rate: float = 100.0, dt: float = 0.01, config=None):
        """
        Codificador Sensorial: Converte estados analógicos do mundo real 
        em sinais (Corrente ou Spikes) para o cérebro (SNN).
        """
        options = (config or {}).get("perception", {}).get("sensor_encoder", {})
        self.max_rate = float(options.get("max_rate", max_rate))
        self.dt = float(options.get("dt", dt))
        self.magnitude_min = float(options.get("magnitude_min", 0.0))
        self.magnitude_max = float(options.get("magnitude_max", 1.0))
        if self.magnitude_max <= self.magnitude_min:
            raise ValueError("magnitude_max precisa ser maior que magnitude_min")

    def _to_tensor(self, data) -> torch.Tensor:
        """Garante que a entrada do ambiente (numpy) seja convertida com segurança."""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=torch.float32)
        return data.float()

    def encode_rate(self, continuous_values, dt: float | None = None) -> torch.Tensor:
        """
        1. Rate Coding (Poisson): Converte valores em probabilidade de disparo (spikes) 
        proporcional à taxa de disparos.
        """
        tensor_values = self._to_tensor(continuous_values)
        
        # Como o ambiente pode enviar valores negativos (ex: direção esquerda/baixo),
        # usamos a magnitude (abs) para calcular a força do disparo (probabilidade).
        magnitude = torch.abs(tensor_values)
        
        # Normalizar pela faixa configurada e limitar a entrada fora da faixa.
        normalized = torch.clamp(
            (magnitude - self.magnitude_min) / (self.magnitude_max - self.magnitude_min),
            0.0, 1.0,
        )
        
        # Gerar probabilidade de pulso baseada no valor
        # Probabilidade de Bernoulli satura em 1 para taxas/dt acima da faixa.
        probabilities = torch.clamp(normalized * self.max_rate * (self.dt if dt is None else dt), 0.0, 1.0)
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
