'''Modulo: experiments/latency_profile.py'''
import time
import torch
from core.spike_engine import SpikeEngine

def benchmark_latency_and_throughput():
    print("="*60)
    print("⏱️ ITEM 26: BENCHMARK DE LATÊNCIA POR PASSO E THROUGHPUT")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de teste: {device}\n")

    # Testa com uma rede de tamanho robusto: [16 -> 2048 -> 4]
    layer_sizes = [16, 2048, 4]
    density = 0.15
    
    print(f"Instanciando rede esparsa {layer_sizes} (Densidade: {density*100}%)...")
    engine = SpikeEngine(layer_sizes=layer_sizes, density=density, device=str(device))
    
    dummy_input = torch.randn(16, device=device)
    
    # Aquecimento (Warm-up do JIT/PyTorch)
    for _ in range(10):
        _ = engine(dummy_input)

    # Parâmetros do Benchmark
    num_steps = 500
    print(f"Executando simulação contínua por {num_steps} passos temporais...\n")

    start_time = time.perf_counter()
    total_spikes_generated = 0

    for _ in range(num_steps):
        spikes_list = engine(dummy_input)
        
        # Conta quantos spikes totais dispararam em todas as camadas neste tick
        tick_spikes = sum(tensor.sum().item() for tensor in spikes_list)
        total_spikes_generated += tick_spikes

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Métricas calculadas
    avg_latency_ms = (total_time / num_steps) * 1000.0
    throughput_steps_per_sec = num_steps / total_time
    throughput_spikes_per_sec = total_spikes_generated / total_time

    print("-" * 60)
    print(f"Tempo Total de Execução    : {total_time:.4f} segundos")
    print(f"Latência Média por Passo   : {avg_latency_ms:.2f} ms / tick")
    print(f"Throughput de Simulação    : {throughput_steps_per_sec:.2f} passos/segundo")
    print(f"Throughput de Spikes       : {throughput_spikes_per_sec:.2f} spikes/segundo")
    print("="*60)
    print("✅ Benchmark de Latência Concluído com Sucesso!")

if __name__ == "__main__":
    benchmark_latency_and_throughput()