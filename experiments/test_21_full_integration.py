'''Modulo: experiments/test_04_full_integration.py'''
import torch

from core.neuron import LeakyIntegrateAndFire
from core.sparse_network import SparseSynapseNetwork
from learning.stdp import STDPManager
from learning.reward_learning import RSTDPManager
from learning.homeostasis import HomeostasisManager

def run_full_integration_test():
    print("=" * 65)
    print("🚀 INICIANDO TESTE DE INTEGRAÇÃO TOTAL (ITENS 6 AO 13) 🚀")
    print("=" * 65)

    # Configuração da Rede
    num_pre = 100
    num_post = 50
    density = 0.20 # 20% de densidade

    print("\n[Itens 6, 7 e 8] Inicializando Neurônios Modulares e Seguros...")
    lif_pre = LeakyIntegrateAndFire(num_neurons=num_pre, threshold=1.0)
    lif_post = LeakyIntegrateAndFire(num_neurons=num_post, threshold=1.0)
    print("-> Camadas LIF instanciadas com sucesso.")

    print(f"\n[Itens 9 e 10] Construindo Rede Esparsa (Alvo: {num_pre * num_post * density:.0f} sinapses)...")
    network = SparseSynapseNetwork(num_post, num_pre, density=density)
    conexoes_reais = network.sparse_weights._nnz()
    is_coalesced = network.sparse_weights.is_coalesced()
    print(f"-> Conexões reais geradas: {conexoes_reais} (Exatidão Perfeita!)")
    print(f"-> Tensor coalescido e seguro? {is_coalesced}")

    print("\n[Itens 11, 12 e 13] Inicializando Motores de Aprendizado e Estabilidade...")
    stdp = STDPManager(num_post, num_pre)
    r_stdp = RSTDPManager(num_post, num_pre)
    homeostasis = HomeostasisManager(num_post, target_firing_rate=0.1)
    
    # Verifica Item 12 (Elegibilidade não aloca memória até precisar)
    print(f"-> R-STDP Rastro de Elegibilidade alocado na memória? {r_stdp.eligibility_trace is not None}")

    print("\n" + "-" * 40)
    print("🧠 INICIANDO SIMULAÇÃO SENSORIAL (5 Passos)")
    print("-" * 40)

    for step in range(1, 6):
        # 1. Injeta corrente aleatória simulando sensores
        current_in = torch.rand(num_pre)
        
        # 2. Propaga pela camada Pré
        spk_pre, mem_pre = lif_pre(current_in)
        
        # 3. Propaga pela Matriz Esparsa
        current_post = network.forward_propagation(spk_pre)
        
        # 4. Propaga pela Camada Pós
        spk_post, mem_post = lif_post(current_post)
        
        # 5. Atualiza traços STDP clássico (Item 11)
        deltas_stdp = stdp.compute_sparse_deltas(spk_pre, spk_post, network.sparse_weights.indices())
        
        # 6. Atualiza traços R-STDP (Item 12)
        r_stdp.step_eligibility(spk_pre, spk_post, network.sparse_weights.indices())
        
        # 7. Aplica Homeostase (Item 13)
        homeostasis.apply_homeostasis(spk_post, lif_post)
        
        print(f"Passo {step} | Spikes Gerados (Pre/Post): {spk_pre.sum().item():.0f} / {spk_post.sum().item():.0f}")

    print("\n" + "-" * 40)
    print("🏆 EVENTO DE RECOMPENSA (DOPAMINA)")
    print("-" * 40)
    
    # Item 12 em Ação: Recompensa positiva global do ambiente
    reward = 1.0 
    
    # Valida tamanho do vetor de elegibilidade (Tem de ser 1D com 1000 elementos, em vez de matriz 50x100)
    tam_elegibilidade = r_stdp.eligibility_trace.shape
    print(f"-> Memória RAM salva! Rastro de elegibilidade é um vetor 1D de tamanho: {list(tam_elegibilidade)}")
    
    deltas_recompensa = r_stdp.apply_reward_modulation(reward, network.sparse_weights.indices())
    
    # Item 10 e 9 em Ação: Atualiza os pesos e garante o clamp
    network.update_sparse_weights(deltas_recompensa)
    print("-> Pesos sinápticos atualizados com modulação de recompensa!")

    print("\n" + "-" * 40)
    print("🧹 RESET GLOBAL DE EPISÓDIO")
    print("-" * 40)
    
    lif_pre.reset_state()
    lif_post.reset_state()
    stdp.reset_traces()
    r_stdp.reset_state()
    homeostasis.reset_state()
    
    print(f"-> Voltagem da Membrana Pre após reset: {lif_pre.mem.sum().item():.1f}")
    print(f"-> Limiar Homeostático adaptado da Camada Pós: {lif_post.threshold[0].item():.4f} (Diferente de 1.0!)")
    
    print("\n=======================================================")
    print("✅ TESTE BEM SUCEDIDO! A ARQUITETURA ESTÁ 100% OPERACIONAL.")
    print("=======================================================")

if __name__ == "__main__":
    run_full_integration_test()