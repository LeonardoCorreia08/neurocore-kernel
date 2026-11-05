'''Modulo: test_05_scale_10k.py'''
import torch
import time
from core.config_loader import load_brain_config
from core.neuron import LeakyIntegrateAndFire
from core.sparse_network import SparseSynapseNetwork

def run_scale_10k_test():
    print("Carregando configuracoes do cerebro artificial...")
    config = load_brain_config()
    
    # A escala deste experimento está no perfil legado da configuração central.
    scale_config = config.get("legacy", {}).get("scale", {})
    neuron_config = config.get("legacy", {}).get("neuron", {})
    target_neurons = scale_config.get("target_neurons", 10_000)
    density = scale_config.get("connectivity_density", 0.15)
    threshold = neuron_config.get("threshold", 0.5)
    beta = neuron_config.get("beta", 0.9)
    
    print(f"Iniciando Teste de Escala — Alvo: {target_neurons} neuronios | Densidade Esparsa: {density * 100}%")
    
    # 1. Instanciar a camada neural de grande escala
    neuron_layer = LeakyIntegrateAndFire(
        num_neurons=target_neurons,
        threshold=threshold,
        decay=beta,
    )
    
    # 2. Instanciar a rede de sinapses esparsas (Matriz W de 10k x 10k esparsa)
    print("Construindo matriz de sinapses esparsas (aguarde alguns segundos)...")
    start_time = time.time()
    
    synapse_network = SparseSynapseNetwork(
        num_posts=target_neurons, 
        num_pres=target_neurons, 
        density=density, 
        weight_scale=0.1
    )
    
    build_duration = time.time() - start_time
    print(f"Rede esparsa construida com sucesso em {build_duration:.2f} segundos!")
    
    # 3. Simular passos de tempo para medir desempenho e propagacao de spikes
    num_steps = 50
    print(f"Executando simulação de {num_steps} passos de tempo para {target_neurons} neuronios...")
    
    neuron_layer.reset_state()
    
    # Criar um vetor de entrada inicial aleatorio (ex: estimulo sensorial esparso)
    x_input_base = torch.rand(target_neurons) * 0.4
    
    total_spikes_recorded = 0
    sim_start_time = time.time()
    
    for t in range(num_steps):
        # Injetar estimulo e calcular corrente via propagacao esparsa
        # Simulamos que os neuronios anteriores disparam com base no input
        spk_pre = (torch.rand(target_neurons) > 0.85).float() # 15% de atividade aleatoria de entrada
        
        # Produto esparso: W * spk_pre
        current = synapse_network.forward_propagation(spk_pre) + x_input_base
        
        # Propagacao pelos neuronios LIF
        spikes, mem = neuron_layer(current)
        
        step_spikes = spikes.sum().item()
        total_spikes_recorded += step_spikes
        
        if (t + 1) % 10 == 0:
            print(f"Passo {t + 1}/{num_steps} concluido | Spikes disparados neste passo: {step_spikes}")

    sim_duration = time.time() - sim_start_time
    print("\n--- Relatorio de Desempenho (NeuroCore 10k) ---")
    print(f"Tempo total de simulacao: {sim_duration:.4f} segundos")
    print(f"Media de spikes por passo: {total_spikes_recorded / num_steps:.2f}")
    print("Teste de escala executado com sucesso!")

if __name__ == "__main__":
    run_scale_10k_test()
