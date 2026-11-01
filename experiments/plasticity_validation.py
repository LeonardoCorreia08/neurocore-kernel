'''Modulo: experiments/plasticity_validation.py'''
import torch
from core.spike_engine import SpikeEngine

def validate_plasticity():
    print("="*70)
    print("🧬 ITENS 29 & 30: VALIDAÇÃO CIENTÍFICA DE STDP E R-STDP (CORRIGIDO)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    engine = SpikeEngine(layer_sizes=[4, 16, 4], density=0.5, device=str(device))
    
    initial_weights = engine.synapses[0].sparse_weights.values().clone()
    print(f"Pesos sinápticos iniciais (amostra): {initial_weights[:5].detach().cpu().numpy()}")

    # 1. Validação STDP (Item 29)
    print("\n[Item 29] Simulando atividade com STDP ativo...")
    dummy_input = torch.tensor([1.0, 0.5, 1.0, 0.0], device=device)
    
    for _ in range(50):
        _ = engine(dummy_input)

    weights_after_stdp = engine.synapses[0].sparse_weights.values().clone()
    stdp_diff = (weights_after_stdp - initial_weights).abs().sum().item()
    print(f"-> Alteração total acumulada por STDP: {stdp_diff:.6f}")

    # 2. Validação R-STDP (Item 30) - Injetando recompensa EM CONJUNTO com novos disparos
    print("\n[Item 30] Simulando episódios com injeção ativa de Dopamina (R-STDP)...")
    
    reward_diff_total = 0.0
    for episode in range(5):
        # Dispara alguns passos
        for _ in range(10):
            _ = engine(dummy_input)
            
        weights_before_reward = engine.synapses[0].sparse_weights.values().clone()
        
        # Injeção direta de Dopamina após uma sequência ativa
        engine.apply_global_reward(reward=1.5)
        
        weights_after_reward = engine.synapses[0].sparse_weights.values().clone()
        ep_diff = (weights_after_reward - weights_before_reward).abs().sum().item()
        reward_diff_total += ep_diff

    print(f"-> Alteração total acumulada por Modulação Dopaminérgica (R-STDP): {reward_diff_total:.6f}")

    print("-" * 70)
    if stdp_diff > 0.0 and reward_diff_total > 0.0:
        print("✅ SUCESSO CIENTÍFICO COMPLETO:")
        print("   - STDP validado com sucesso (Plasticidade temporal ativa).")
        print("   - R-STDP validado com sucesso (Modulação por Dopamina comprovada).")
    else:
        print("⚠️ Atenção: Verifique o fluxo de elegibilidade.")
    print("="*70)

if __name__ == "__main__":
    validate_plasticity()