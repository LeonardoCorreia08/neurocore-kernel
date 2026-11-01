'''Modulo: test_09_memory_modules.py'''
import torch
from memory.long_term import LongTermMemoryManager
from memory.associative import AssociativeMemoryNetwork
from memory.consolidation import ConsolidationEngine
from core.synapse import SparseSynapseManager

def run_memory_modules_test():
    print("Iniciando Teste dos Subsistemas de Memória (NeuroCore)...")
    
    # 1. Testar Memória de Longo Prazo (LTM)
    ltm_manager = LongTermMemoryManager(storage_dir="memory/storage")
    dummy_state = {"weights_sample": torch.tensor([0.5, -0.2, 0.8]), "epoch": 10}
    ltm_manager.store_memory_snapshot("neurocore_snapshot_test", dummy_state)
    loaded_state = ltm_manager.load_memory_snapshot("neurocore_snapshot_test")
    print(f"[LTM Test] Estado carregado com sucesso: {loaded_state}")
    
    # 2. Testar Memória Associativa (Hebriana)
    vector_dim = 4
    assoc_net = AssociativeMemoryNetwork(vector_dim=vector_dim)
    pattern_a = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pattern_b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    assoc_net.encode_association(pattern_a, pattern_b)
    recalled_pattern = assoc_net.recall(pattern_a)
    print(f"[Associative Test] Padrao recuperado a partir de A: {recalled_pattern}")
    
    # 3. Testar Motor de Consolidação e Esquecimento Ativo
    print("[Consolidation Test] Criando matriz esparsa para teste de poda...")
    synapse_manager = SparseSynapseManager(num_posts=100, num_pres=100, density=0.2)
    
    consolidation_engine = ConsolidationEngine(pruning_threshold=0.05)
    filtered_weights = consolidation_engine.apply_active_forgetting(synapse_manager.sparse_weights)
    print(f"[Consolidation Test] Poda aplicada com sucesso. Sinapses ativas restantes: {filtered_weights._nnz()}")
    
    print("Todos os testes dos subsistemas de memoria foram concluidos com sucesso!")

if __name__ == "__main__":
    run_memory_modules_test()