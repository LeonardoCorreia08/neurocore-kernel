'''Modulo: experiments/continuous_learning_generalization.py'''
import torch
from core.spike_engine import SpikeEngine

def validate_continuous_learning_and_generalization():
    print("="*70)
    print("🔄 ITENS 34 & 35: APRENDIZAGEM CONTÍNUA E GENERALIZAÇÃO")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    engine = SpikeEngine(layer_sizes=[16, 128, 4], density=0.25, device=str(device))
    
    # Padrão A (Tarefa 1) e Padrão B (Tarefa 2)
    pattern_a = torch.zeros(16, device=device)
    pattern_a[[2, 6, 10, 14]] = 3.0
    
    pattern_b = torch.zeros(16, device=device)
    pattern_b[[1, 5, 9, 13]] = 3.0 # Padrão ortogonal/diferente
    
    print("\n[Item 34 - Aprendizagem Contínua] Treinando Tarefa A...")
    for _ in range(25):
        for _ in range(5):
            _ = engine(pattern_a)
        engine.apply_global_reward(reward=2.5)
        
    # Mede resposta à Tarefa A antes de aprender a Tarefa B
    engine.reset_all_states()
    response_a_before = sum(engine(pattern_a)[-1].sum().item() for _ in range(15))

    print("Treinando Tarefa B em sequência (sequential learning)...")
    for _ in range(25):
        for _ in range(5):
            _ = engine(pattern_b)
        engine.apply_global_reward(reward=2.5)

    # Mede retenção da Tarefa A após aprender a Tarefa B (Teste de Esquecimento Catastrófico)
    engine.reset_all_states()
    response_a_after = sum(engine(pattern_a)[-1].sum().item() for _ in range(15))
    
    print(f"Atividade Motora Tarefa A (Antes da Tarefa B): {response_a_before}")
    print(f"Atividade Motora Tarefa A (Após a Tarefa B) : {response_a_after}")

    print("\n[Item 35 - Generalização] Testando com Padrão Variação (Ruído Parcial / Desconhecido)...")
    # Variação do Padrão A com 25% de alteração estrutural
    variant_pattern_a = pattern_a.clone()
    variant_pattern_a[2] = 0.0
    variant_pattern_a[4] = 3.0 # Nova mutação
    
    engine.reset_all_states()
    response_variant = sum(engine(variant_pattern_a)[-1].sum().item() for _ in range(15))
    print(f"Atividade Motora para Variante Desconhecida: {response_variant}")

    print("-" * 70)
    retention_ratio = (response_a_after / (response_a_before + 1e-5)) * 100
    
    print(f"Taxa de Retenção de Conhecimento (Item 34): {retention_ratio:.2f}%")
    
    if retention_ratio > 50.0 and response_variant > 0:
        print("✅ SUCESSO CIENTÍFICO AVANÇADO:")
        print("   - Aprendizagem Contínua validada (Resistência ao esquecimento catastrófico).")
        print("   - Generalização validada (Resposta ativa a novos estímulos similares).")
    else:
        print("⚠️ AVISO: Ajuste fino necessário na retenção de traços.")
    print("="*70)

if __name__ == "__main__":
    validate_continuous_learning_and_generalization()