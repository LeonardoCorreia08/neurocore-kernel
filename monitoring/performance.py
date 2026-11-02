'''Modulo: performance.py'''


def _settings(config):
    if not config:
        return {}
    return config.get("monitoring", config).get("performance", {})


class PerformanceEvaluator:
    def __init__(self, joules_per_spike: float = 1e-9, joules_per_sop: float = 1e-12, config: dict = None):
        settings = _settings(config)
        self.joules_per_spike = settings.get("joules_per_spike", joules_per_spike)
        self.joules_per_sop = settings.get("joules_per_sop", joules_per_sop)
        self.metrics = {}

    def evaluate_energy_cost(self, active_spikes_count: int, synaptic_operations: int) -> float:
        """Estima o custo energetico com base no numero de spikes e operacoes sinapticas (SOPs)."""
        # Preserva o modelo linear estimado existente; valores agora são configuráveis.
        return (active_spikes_count * self.joules_per_spike) + (synaptic_operations * self.joules_per_sop)

    def compute_success_rate(self, successes: int, total_episodes: int) -> float:
        """Calcula a taxa de acerto percentual do agente no ambiente."""
        if total_episodes == 0:
            return 0.0
        return (successes / total_episodes) * 100.0
