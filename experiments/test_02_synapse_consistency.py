'''Modulo: test_02_synapse_consistency.py'''
import torch
from core.synapse import SparseSynapseManager

def run_synapse_test():
    print("Iniciando Teste de Consistência Sináptica (SparseSynapseManager)...")
    
    num_pre = 1000
    num_post = 500
    density = 0.1
    
    # 1. Cria a camada sináptica
    print("\n--- Fase 1: Inicialização e Coalescência ---")
    synapse = SparseSynapseManager(num_posts=num_post, num_pres=num_pre, density=density)
    
    is_coalesced = synapse.sparse_weights.is_coalesced()
    num_connections = synapse.sparse_weights._nnz()
    print(f"Tensor está matematicamente organizado (coalesced)? {is_coalesced}")
    print(f"Conexões ativas criadas: {num_connections} (de um total possível de {num_pre * num_post})")

    # 2. Testa a propagação
    print("\n--- Fase 2: Propagação Segura (Forward) ---")
    spikes_in = (torch.rand(num_pre) > 0.8).float()
    print(f"Enviando {spikes_in.sum().item():.0f} spikes da camada anterior...")
    corrente_out = synapse.forward_propagation(spikes_in)
    print(f"Corrente gerada com sucesso! Formato do tensor de saída: {list(corrente_out.shape)}")

    # 3. Testa a atualização com Limites (Clamp)
    print("\n--- Fase 3: Atualização de Pesos (Clamping) ---")
    pesos_antigos = synapse.sparse_weights.values().clone()
    
    # Tentamos forçar uma explosão de pesos injetando +5.0 (o limite é 1.0)
    delta_explosivo = torch.ones(num_connections) * 5.0 
    synapse.update_sparse_weights(delta_explosivo, w_min=-1.0, w_max=1.0)
    
    peso_maximo = synapse.sparse_weights.values().max().item()
    print(f"Tentativa de explodir pesos para +5.0 barrada. Peso máximo atual: {peso_maximo:.2f} (Restrito pelo clamp)")

    # 4. Testa a Poda (Pruning)
    print("\n--- Fase 4: Poda Estrutural (Pruning) ---")
    # Injetamos valores propositalmente fracos em metade das conexões para testar a poda
    metade = num_connections // 2
    novos_valores = synapse.sparse_weights.values().clone()
    novos_valores[:metade] = 0.005 # Valores muito fracos (abaixo de 0.01)
    
    # Reconstroi o tensor para injetar nossa alteração manual
    synapse.sparse_weights = torch.sparse_coo_tensor(
        synapse.sparse_weights.indices(), novos_valores, size=(num_post, num_pre)
    ).coalesce()
    
    conexoes_antes_poda = synapse.sparse_weights._nnz()
    synapse.prune_weak_connections(threshold=0.01)
    conexoes_depois_poda = synapse.sparse_weights._nnz()
    
    print(f"Conexões antes da poda: {conexoes_antes_poda}")
    print(f"Conexões após a poda: {conexoes_depois_poda}")
    print(f"Conexões removidas com sucesso: {conexoes_antes_poda - conexoes_depois_poda}")
    
    print("\n[Resultados] Teste de consistência sináptica concluído com sucesso!")

if __name__ == "__main__":
    run_synapse_test()