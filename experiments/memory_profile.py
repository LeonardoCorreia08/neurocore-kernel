'''Modulo: experiments/memory_profile.py'''
import torch
import psutil
import os
from core.spike_engine import SpikeEngine

def profile_network_scaling():
    print("="*60)
    print("🔬 ITEM 25: PROFILING DE RAM E VRAM POR ESCALA DE NEURÓNIOS")
    print("="*60)

    # Coleta de processo atual para medir RAM do sistema
    process = psutil.Process(os.getpid())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo ativo para profiling: {device}\n")

    # Diferentes escalas de teste para a camada oculta
    scales = [32, 128, 512, 2048, 8192]
    
    print(f"{'Neurónios Ocultos':<20} | {'RAM Sistema (MB)':<20} | {'VRAM GPU (MB)':<20}")
    print("-" * 68)

    for size in scales:
        # Limpa o cache da GPU antes de cada teste
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Mede RAM antes de instanciar
        ram_before = process.memory_info().rss / (1024 * 1024)

        # Instancia uma rede com a escala atual: [Sensorial(16) -> Oculta(size) -> Motora(4)]
        try:
            engine = SpikeEngine(layer_sizes=[16, size, 4], density=0.15, device=str(device))
            
            # Executa um ciclo rápido para forçar alocação de buffers de traços e elegibilidade
            dummy_input = torch.randn(16, device=device)
            _ = engine(dummy_input)
            
            # Mede RAM após instanciar e rodar
            ram_after = process.memory_info().rss / (1024 * 1024)
            ram_consumed = ram_after - ram_before

            # Mede VRAM se houver GPU
            vram_consumed = 0.0
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.memory_allocated()
                vram_consumed = vram_bytes / (1024 * 1024)

            print(f"{size:<20} | {ram_consumed:<20.2f} | {vram_consumed:<20.2f}")

            # Libera o motor da memória para o próximo loop
            del engine
            
        except RuntimeError as e:
            print(f"{size:<20} | ERRO DE MEMÓRIA (OOM) OU LIMITE ATINGIDO")
            break

    print("="*60)
    print("✅ Profiling de Memória Concluído com Sucesso!")

if __name__ == "__main__":
    profile_network_scaling()