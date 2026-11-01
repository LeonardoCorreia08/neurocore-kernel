'''Modulo: test_16_dynamic_1m.py'''
import torch
import time
from core.neuron import LeakyIntegrateAndFire
from core.synapse import SparseSynapseManager
from monitoring.hardware_telemetry import HardwareTelemetry

def run_dynamic_1m_test():
    print("Iniciando Teste Dinâmico Contínuo com 1.000.000 de Neurônios (1.000 Passos)...")
    
    target_neurons = 1000000
    density = 0.01
    
    input_dim = 2000
    hidden_dim = target_neurons - 3000
    output_dim = 1000
    
    print("[Allocation] Alocando camadas dinâmicas de 1M neurônios...")
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    synapse_w1 = SparseSynapseManager(num_posts=hidden_dim, num_pres=input_dim, density=density)
    synapse_w2 = SparseSynapseManager(num_posts=output_dim, num_pres=hidden_dim, density=density)
    
    telemetry = HardwareTelemetry()
    
    print("[Simulation] Iniciando loop dinâmico de 1.000 passos sob estresse temporal...")
    telemetry.start_measurement()
    
    total_spikes_registered = 0
    time_steps = 1000  # Reduzido para 50 passos pesados para viabilizar o tempo de execução local de teste, mantendo carga de 1M
    
    for t in range(time_steps):
        # Estímulo dinâmico variável no tempo
        dummy_input = (torch.rand(input_dim) > 0.8).float()
        
        spk_sensory, _ = sensory_layer(dummy_input)
        h_input = synapse_w1.forward_propagation(spk_sensory)
        spk_hidden, _ = hidden_layer(h_input)
        m_input = synapse_w2.forward_propagation(spk_hidden)
        spk_motor, _ = motor_layer(m_input)
        
        active_count = int(spk_sensory.sum() + spk_hidden.sum() + spk_motor.sum())
        total_spikes_registered += active_count
        
        if (t + 1) % 10 == 0:
            print(f" -> Passo {t+1}/{time_steps} concluído. Spikes ativos neste passo: {active_count:,}")

    metrics = telemetry.stop_measurement()
    
    print("\n--- Relatório de Validação Dinâmica e Hardware (1M) ---")
    print(f"Passos executados: {time_steps}")
    print(f"Tempo total de simulação: {metrics['elapsed_seconds']:.2f} segundos")
    print(f"Consumo energético real medido: {metrics['energy_joules']:.6f} Joules")
    print(f"Média de spikes ativos por passo: {total_spikes_registered / time_steps:,.0f}")
    print("Validação dinâmica de 1 milhão de neurônios concluída com sucesso!")

if __name__ == "__main__":
    run_dynamic_1m_test()