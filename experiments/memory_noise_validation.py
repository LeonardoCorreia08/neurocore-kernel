'''Modulo: experiments/memory_noise_validation.py'''
import torch
from core.spike_engine import SpikeEngine

def get_temporal_response(engine, pattern, num_ticks=20):
    """Acumula a resposta da camada motora ao longo de vários ticks temporais."""
    engine.reset_all_states()
    accumulated_spikes = None
    
    for _ in range(num_ticks):
        spikes = engine(pattern)
        motor_spikes = spikes[-1].detach().cpu()
        if accumulated_spikes is None:
            accumulated_spikes = motor_spikes
        else:
            accumulated_spikes += motor_spikes
            
    return accumulated_spikes

def validate_memory_with_noise():
    print("="*70)
    print("🧠 ITEM 31: VALIDAÇÃO DE MEMÓRIA E RECUPERAÇÃO COM RUÍDO (TEMPORAL)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    engine = SpikeEngine(layer_sizes=[16, 128, 16], density=0.25, device=str(device))
    
    # Padrão de memória limpo (16 bits)
    memory_pattern = torch.zeros(16, device=device)
    memory_pattern[[2, 5, 8, 11, 14]] = 1.0
    
    print("Treinando o cérebro para consolidar o engrama...")
    
    # Fase de consolidação com reforço dopaminérgico
    for epoch in range(30):
        for _ in range(10):
            _ = engine(memory_pattern * 2.5)
        engine.apply_global_reward(reward=2.0)

    # 1. Resposta de referência ao padrão limpo (acumulada em 20 ticks)
    clean_response = get_temporal_response(engine, memory_pattern * 2.5, num_ticks=20)
    
    # 2. Injetar ruído no padrão de entrada (30% de corrupção)
    print("Aplicando padrão com 30% de ruído estocástico na entrada...")
    noisy_pattern = memory_pattern.clone()
    noise_mask = torch.rand(16, device=device) < 0.3
    noisy_pattern[noise_mask] = 1.0 - noisy_pattern[noise_mask]
    
    noisy_response = get_temporal_response(engine, noisy_pattern * 2.5, num_ticks=20)
    
    # 3. Calcular similaridade
    # Evita divisão por zero caso o vetor seja inteiramente zero
    if clean_response.sum() == 0 or noisy_response.sum() == 0:
        similarity = 0.0
    else:
        similarity = torch.nn.functional.cosine_similarity(clean_response.unsqueeze(0), noisy_response.unsqueeze(0)).item()

    print("-" * 70)
    print(f"Similaridade da Resposta Temporal (Limpo vs Ruidoso): {similarity * 100:.2f}%")
    
    if similarity > 0.65:
        print("✅ SUCESSO: O cérebro recuperou a memória funcional sob ruído estocástico!")
    else:
        print("⚠️ AVISO: Similaridade abaixo do esperado.")
    print("="*70)

if __name__ == "__main__":
    validate_memory_with_noise()