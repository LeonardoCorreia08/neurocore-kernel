'''Modulo: experiments/attention_decision_validation.py'''
import torch
from core.spike_engine import SpikeEngine

def validate_attention_and_decision():
    print("="*70)
    print("🎯 ITENS 32 & 33: VALIDAÇÃO DE ATENÇÃO SELETIVA E TOMADA DE DECISÃO (POSITIVO)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    engine = SpikeEngine(layer_sizes=[16, 128, 4], density=0.3, device=str(device))
    
    target_stimulus = torch.zeros(16, device=device)
    target_stimulus[[3, 7, 12]] = 3.5  # Alvo relevante
    
    distractor_stimulus = torch.zeros(16, device=device)
    distractor_stimulus[[0, 5, 14]] = 3.5 # Distrator (mesma intensidade, mas irrelevante)
    
    print("\n[Item 32 - Atenção] Treinando o cérebro a associar reforço apenas ao Alvo...")
    
    # Treino focado em reforçar o Alvo e ignorar o Distrator
    for epoch in range(35):
        # Apresenta o alvo e aplica forte reforço dopaminérgico positivo
        for _ in range(10):
            _ = engine(target_stimulus)
        engine.apply_global_reward(reward=3.0)
        
        # Apresenta o distrator sem recompensa (neutralidade)
        for _ in range(10):
            _ = engine(distractor_stimulus)
        # Sem recompensa (reward = 0.0) para o distrator

    print("\n[Item 33 - Tomada de Decisão] Testando a escolha motora sob estímulos concorrentes...")
    
    # Testa a resposta ao Alvo
    engine.reset_all_states()
    target_spikes_total = torch.zeros(4)
    for _ in range(20):
        spk = engine(target_stimulus)[-1].detach().cpu()
        target_spikes_total += spk
        
    # Testa a resposta ao Distrator
    engine.reset_all_states()
    distractor_spikes_total = torch.zeros(4)
    for _ in range(20):
        spk = engine(distractor_stimulus)[-1].detach().cpu()
        distractor_spikes_total += spk

    target_total_activity = target_spikes_total.sum().item()
    distractor_total_activity = distractor_spikes_total.sum().item()

    print(f"Volume de Spikes (Alvo)      : {target_total_activity}")
    print(f"Volume de Spikes (Distrator) : {distractor_total_activity}")

    print("-" * 70)
    if target_total_activity > distractor_total_activity and target_total_activity > 0:
        print("✅ SUCESSO COGNITIVO COMPLETO:")
        print("   - Atenção Seletiva validada (O cérebro priorizou o alvo treinado).")
        print("   - Tomada de Decisão validada (Maior atividade motora para a meta).")
    else:
        print("⚠️ AVISO: O cérebro precisa de maior diferenciação de padrões.")
    print("="*70)

if __name__ == "__main__":
    validate_attention_and_decision()