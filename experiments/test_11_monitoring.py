'''Modulo: test_11_monitoring.py'''
import torch
from monitoring.spike_visualizer import SpikeVisualizer
from monitoring.activity_monitor import ActivityMonitor
from monitoring.performance import PerformanceEvaluator

def run_monitoring_test():
    print("Iniciando Teste do Módulo de Avaliação e Telemetria (NeuroCore)...")
    
    # 1. Testar Visualizador de Spikes
    visualizer = SpikeVisualizer()
    for _ in range(50):
        dummy_spikes = (torch.rand(100) > 0.8).float()
        visualizer.record_spikes(dummy_spikes)
    print(f"[Visualizer] Spikes registrados no historico: {len(visualizer.spike_history)} passos.")
    
    # 2. Testar Monitor de Atividade e Homeostase
    activity_monitor = ActivityMonitor(target_firing_rate=15.0)
    rates = activity_monitor.compute_firing_rates(visualizer.spike_history, dt=0.01)
    homeostasis_dev = activity_monitor.check_homeostasis(rates)
    print(f"[Activity Monitor] Frequencia media calculada. Desvio homeostatico: {homeostasis_dev.item():.2f}")
    
    # 3. Testar Avaliador de Desempenho e Energia
    evaluator = PerformanceEvaluator()
    energy = evaluator.evaluate_energy_cost(active_spikes_count=450, synaptic_operations=15000)
    success_rate = evaluator.compute_success_rate(successes=8, total_episodes=10)
    print(f"[Performance Evaluator] Custo energetico estimado: {energy:.8f} Joules")
    print(f"[Performance Evaluator] Taxa de acerto: {success_rate}%")
    
    print("Todos os testes do Módulo 5 (Avaliação e Telemetria) foram concluídos com sucesso!")

if __name__ == "__main__":
    run_monitoring_test()