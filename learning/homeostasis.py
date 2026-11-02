'''Modulo: learning/homeostasis.py'''
import torch
import torch.nn as nn

from core.config_loader import get_config_value


class HomeostasisManager(nn.Module):
    def __init__(self, num_neurons: int, target_firing_rate: float = 0.05, tau_avg: float = 100.0, adjustment_rate: float = 0.01, threshold_min: float = 0.1, threshold_max: float = 5.0, config: dict | None = None):
        """
        Garante a estabilidade a longo prazo da rede, impedindo que neurônios
        dominem a atividade (epilepsia) ou fiquem completamente ociosos.
        """
        super().__init__()
        self.target_rate = get_config_value(config, "learning.homeostasis.target_firing_rate", target_firing_rate)
        self.adjustment_rate = get_config_value(config, "learning.homeostasis.adjustment_rate", adjustment_rate)
        self.threshold_min = get_config_value(config, "learning.homeostasis.threshold_min", threshold_min)
        self.threshold_max = get_config_value(config, "learning.homeostasis.threshold_max", threshold_max)
        tau_avg = get_config_value(config, "learning.homeostasis.tau_avg", tau_avg)

        # Constante de decaimento da média móvel
        self.register_buffer('decay', torch.exp(torch.tensor(-1.0 / tau_avg)))

        # Buffer de rastreamento: Média de disparos de cada neurônio
        self.register_buffer('avg_firing_rate', torch.zeros(num_neurons))

    def reset_state(self):
        """Limpa a média de disparos para um novo experimento isolado."""
        self.avg_firing_rate.zero_()

    def apply_homeostasis(self, spikes: torch.Tensor, neuron_layer: nn.Module):
        """
        Mede a atividade atual e ajusta o limiar (threshold) diretamente na camada LIF.
        Deve ser chamado a cada passo de tempo após o forward da camada.
        """
        spikes = spikes.view(-1)

        # 1. Atualiza a média móvel de disparos
        self.avg_firing_rate = self.avg_firing_rate * self.decay + spikes * (1.0 - self.decay)

        # 2. Calcula o erro em relação à taxa biológica alvo
        rate_error = self.avg_firing_rate - self.target_rate

        # 3. Transforma o threshold num Tensor caso ele ainda seja um float padrão do __init__
        if not isinstance(neuron_layer.threshold, torch.Tensor):
            neuron_layer.threshold = torch.ones_like(self.avg_firing_rate) * neuron_layer.threshold
            neuron_layer.threshold = neuron_layer.threshold.to(spikes.device)

        # 4. Ajuste Dinâmico: Aumenta o limiar se disparou muito, reduz se disparou pouco
        neuron_layer.threshold = neuron_layer.threshold + (rate_error * self.adjustment_rate)

        # 5. Clamp de segurança, com limites configuráveis
        neuron_layer.threshold = torch.clamp(neuron_layer.threshold, min=self.threshold_min, max=self.threshold_max)
