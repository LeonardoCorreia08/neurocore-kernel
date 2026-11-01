'''Modulo: experiments/neural_dynamics_analysis.py'''
import torch
import numpy as np
from core.spike_engine import SpikeEngine

def analyze_neural_dynamics():
    print("="*65)
    print("📊 ITEM 28: ANÁLISE DE DINÂMICA NEURAL (ESTABILIDADE & FREQUÊNCIA)")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Arquitetura de teste: [Sensorial(16) -> Oculta(1024) -> Motora(4)]
    layer_sizes = [16, 1024, 4]
    engine = SpikeEngine(layer_sizes=layer_sizes, density=0.15, device=str(device))
    
    num_steps = 300
    print(f"Simulando dinâmica por {num_steps} passos temporais...")

    # Coordenadores de spikes por camada ao longo do tempo
    layer_spike_history = {i: [] for i in range(len(layer_sizes))}

    # Simulação contínua com ruído sensorial estocástico
    for step in range(num_steps):
        # Entrada com ruído poissoniano simulado
        sensory_input = torch.rand(16, device=device) * 1.5
        
        spikes = engine(sensory_input)
        
        for layer_idx, spk in enumerate(spikes):
            layer_spike_history[layer_idx].append(spk.detach().cpu().numpy())

    print("\n" + "-"*65)
    print(f"{'Camada':<15} | {'Total Spikes':<15} | {'Taxa Média (Hz/neurónio)':<25} | {'Estabilidade':<12}")
    print("-" * 65)

    for layer_idx, size in enumerate(layer_sizes):
        history = np.array(layer_spike_history[layer_idx]) # shape: [steps, neurons]
        total_spikes = np.sum(history)
        
        # Taxa de disparo média por neurónio por passo (assumindo 1 tick = 1ms -> taxa em kHz/Hz)
        firing_rate_per_neuron = total_spikes / (num_steps * size)
        
        # Coeficiente de variação / Estabilidade temporal (desvio padrão dos disparos por passo)
        step_spike_counts = np.sum(history, axis=1)
        stability_index = np.std(step_spike_counts) / (np.mean(step_spike_counts) + 1e-5)
        
        status = "Estável" if stability_index < 1.5 else "Bursting/Volátil"
        
        print(f"Camada {layer_idx} ({size}n) | {int(total_spikes):<15,d} | {firing_rate_per_neuron * 1000.0:<25.2f} Hz | {status:<12}")

    print("="*65)
    print("✅ Análise de Dinâmica Neural Concluída com Sucesso!")

if __name__ == "__main__":
    analyze_neural_dynamics()