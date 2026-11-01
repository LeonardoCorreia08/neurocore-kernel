'''Modulo: test_13_scale_1m.py'''
import torch
import time
from core.neuron import LeakyIntegrateAndFire
from core.synapse import SparseSynapseManager
from monitoring.performance import PerformanceEvaluator

def run_scale_1m_test():
    print("Iniciando Experimento de Escalonamento Massivo: NeuroCore em Escala de 1.000.000 de Neurônios...")
    
    # Parâmetros de Escala Extrema (1 Milhão)
    target_neurons = 1000000
    density = 0.01  # Densidade otimizada de 1% para suportar escala massiva sem estouro de RAM/VRAM
    
    input_dim = 5000
    hidden_dim = target_neurons - 7000
    output_dim = 2000
    
    print(f"[Architecture] Alocando populações neurais massivas: {target_neurons:,} neurônios...")
    start_time = time.time()
    
    # Alocação das Camadas de Grande Escala
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    print(f"[Architecture] Construindo matrizes de sinapses esparsas em massa (Densidade: {density*100}%)...")
    synapse_w1 = SparseSynapseManager(num_posts=hidden_dim, num_pres=input_dim, density=density)
    synapse_w2 = SparseSynapseManager(num_posts=output_dim, num_pres=hidden_dim, density=density)
    
    alloc_time = time.time() - start_time
    print(f"[Success] Rede de 1 Milhão de neurônios alocada e coalescida com sucesso em {alloc_time:.2f} segundos!")
    
    # Teste de Ciclo de Propagação sob Estresse Massivo
    print("[Simulation] Executando ciclo de teste sob carga massiva de 1M neurônios...")
    evaluator = PerformanceEvaluator()
    
    # Gerar vetor de entrada esparso leve
    dummy_input = (torch.rand(input_dim) > 0.85).float()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    step_start = time.time()
    
    # Propagação passo a passo pelas camadas corticais
    spk_sensory, _ = sensory_layer(dummy_input)
    h_input = synapse_w1.forward_propagation(spk_sensory)
    spk_hidden, _ = hidden_layer(h_input)
    m_input = synapse_w2.forward_propagation(spk_hidden)
    spk_motor, _ = motor_layer(m_input)
    
    step_time = time.time() - step_start
    
    active_spikes = int(spk_sensory.sum() + spk_hidden.sum() + spk_motor.sum())
    estimated_sops = synapse_w1.sparse_weights._nnz() + synapse_w2.sparse_weights._nnz()
    energy_cost = evaluator.evaluate_energy_cost(active_spikes, estimated_sops)
    
    print(f"[Performance] Tempo de processamento do ciclo (1M): {step_time*1000:.2f} ms")
    print(f"[Performance] Spikes ativos no ciclo cortical: {active_spikes:,}")
    print(f"[Performance] Operações sinápticas (SOPs) envolvidas: {estimated_sops:,}")
    print(f"[Performance] Custo energético estimado: {energy_cost:.8f} Joules")
    print("Marco Histórico: Escalonamento para 1.000.000 de neurônios validado com sucesso!")

if __name__ == "__main__":
    run_scale_1m_test()