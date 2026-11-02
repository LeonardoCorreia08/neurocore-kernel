'''Modulo: activity_monitor.py'''
import torch


def _section(config, name):
    if not config:
        return {}
    return config.get("monitoring", config).get(name, {})


class ActivityMonitor:
    def __init__(self, target_firing_rate: float = 20.0, config: dict = None):
        settings = _section(config, "activity")
        self.target_firing_rate = settings.get("target_firing_rate", target_firing_rate)
        self.dt = settings.get("dt", 0.01)

    def compute_firing_rates(self, spike_history: list[torch.Tensor], dt: float = None) -> torch.Tensor:
        """Calcula a frequencia media de disparo (Hz) por neuronio com base no historico."""
        if not spike_history:
            return torch.tensor(0.0)

        dt = self.dt if dt is None else dt
        stacked = torch.stack(spike_history)  # [tempo, neuronios]
        total_time = stacked.shape[0] * dt
        spike_counts = torch.sum(stacked, dim=0)
        firing_rates = spike_counts / total_time
        return firing_rates

    def check_homeostasis(self, firing_rates: torch.Tensor) -> torch.Tensor:
        """Avalia o desvio da taxa de disparo em relacao ao alvo homeostatico."""
        mean_rate = torch.mean(firing_rates)
        deviation = mean_rate - self.target_firing_rate
        return deviation
