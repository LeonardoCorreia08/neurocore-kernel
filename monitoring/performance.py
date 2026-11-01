'''Modulo: performance.py'''
import torch

class PerformanceEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_energy_cost(self, active_spikes_count: int, synaptic_operations: int) -> float:
        """Estima o custo energetico com base no numero de spikes e operacoes sinapticas (SOPs)."""
        # Constantes simuladas de consumo por pulso e operacao sinaptica
        joules_per_spike = 1e-9
        joules_per_sop = 1e-12
        
        total_energy = (active_spikes_count * joules_per_spike) + (synaptic_operations * joules_per_sop)
        return total_energy

    def compute_success_rate(self, successes: int, total_episodes: int) -> float:
        """Calcula a taxa de acerto percentual do agente no ambiente."""
        if total_episodes == 0:
            return 0.0
        return (successes / total_episodes) * 100.0