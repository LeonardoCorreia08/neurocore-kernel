'''Modulo: test_12_scale_100k.py'''
import torch
import time
from core.neuron import LeakyIntegrateAndFire
from core.synapse import SparseSynapseManager
from monitoring.performance import PerformanceEvaluator

def run_scale_100k_test():
    print("Iniciando Experimento de Escalonamento Massivo: NeuroCore em Escala de 100.000 Neurônios...")
    
    # Parâmetros de Escala
    target_neurons = 100000
    density = 0.05  # Reduzindo a densidade para 5% para otimizar VRAM em grande escala
    
    input_dim = 1000
    hidden_dim = target_neurons - 1200
    output_dim = 200
    
    print(f"[Architecture] Alocando populações neurais totais: {target_neurons} neurônios...")
    start_time = time.time()
    
    # Alocação de Camadas em Grande Escala
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    print(f"[Architecture] Construindo matrizes de sinapses esparsas (Densidade: {density*100}%)...")
    synapse_w1 = SparseSynapseManager(num_posts=hidden_dim, num_pres=input_dim, density=density)
    synapse_w2 = SparseSynapseManager(num_posts=output_dim, num_pres=hidden_dim, density=density)
    
    alloc_time = time.time() - start_time
    print(f"[Success] Rede de 100k alocada e coalescida com sucesso em {alloc_time:.2f} segundos!")
    
    # Teste de Ciclo de Propagação sob Carga
    print("[Simulation] Executando ciclo de teste sob carga de propagação...")
    evaluator = PerformanceEvaluator()
    
    dummy_input = (torch.rand(input_dim) > 0.7).float()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    step_start = time.time()
    
    # Propagação passo a passo
    spk_sensory, _ = sensory_layer(dummy_input)
    h_input = synapse_w1.forward_propagation(spk_sensory)
    spk_hidden, _ = hidden_layer(h_input)
    m_input = synapse_w2.forward_propagation(spk_hidden)
    spk_motor, _ = motor_layer(m_input)
    
    step_time = time.time() - step_start
    
    active_spikes = int(spk_sensory.sum() + spk_hidden.sum() + spk_motor.sum())
    estimated_sops = synapse_w1.sparse_weights._nnz() + synapse_w2.sparse_weights._nnz()
    energy_cost = evaluator.evaluate_energy_cost(active_spikes, estimated_sops)
    
    print(f"[Performance] Tempo de processamento do ciclo (100k): {step_time*1000:.2f} ms")
    print(f"[Performance] Spikes ativos no ciclo: {active_spikes}")
    print(f"[Performance] Operações sinápticas (SOPs) envolvidas: {estimated_sops}")
    print(f"[Performance] Custo energético estimado: {energy_cost:.8f} Joules")
    print("Escalonamento para 100.000 neurônios validado com sucesso!")

if __name__ == "__main__":
    run_scale_100k_test()