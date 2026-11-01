'''Modulo: experiments/million_neuron_scale_test.py'''
import time
import torch
import psutil
import os
from core.sparse_network import SparseSynapseNetwork

def scale_benchmark():
    print("="*75)
    print("🔬 EXPERIMENTO CIENTÍFICO: ESCALA MASSIVA (10K, 100K, 1M NEURÔNIOS)")
    print("="*75)

    process = psutil.Process(os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de Execução: {device}\n")

    # Escalas científicas estipuladas
    # Camada de entrada fixa em 100, camada oculta escalada, saída em 10
    scales = [
        {"name": "10 Mil Neurônios", "hidden": 10_000, "density": 0.05},
        {"name": "100 Mil Neurônios", "hidden": 100_000, "density": 0.01}, # Densidade reduzida para evitar OOM geométrico
        {"name": "1 Milhão de Neurônios", "hidden": 1_000_000, "density": 0.002} # 0.2% de densidade para 1M
    ]

    print(f"{'Escala':<22} | {'Neurônios (Hidden)':<18} | {'Sinapses Ativas (K)':<20} | {'RAM (MB)':<12} | {'Criação (s)':<12}")
    print("-" * 92)

    for sc in scales:
        hidden_size = sc["hidden"]
        density = sc["density"]
        input_size = 100

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ram_before = process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()

        try:
            # Instancia apenas a camada sináptica principal para medir o peso bruto da escala
            synapse_net = SparseSynapseNetwork(num_posts=hidden_size, num_pres=input_size, density=density)
            
            if device.type == "cuda":
                synapse_net = synapse_net.to(device)

            end_time = time.perf_counter()
            ram_after = process.memory_info().rss / (1024 * 1024)
            
            ram_consumed = ram_after - ram_before
            creation_time = end_time - start_time
            num_synapses = synapse_net.sparse_weights._nnz()

            print(f"{sc['name']:<22} | {hidden_size:<18,d} | {num_synapses:<20,d} | {ram_consumed:<12.2f} | {creation_time:<12.4f}")

            # Teste rápido de propagação (Forward pass)
            dummy_spike = torch.zeros(input_size, device=device)
            dummy_spike[torch.randint(0, input_size, (int(input_size * 0.1),))] = 1.0 # 10% firing rate
            
            t_fwd_start = time.perf_counter()
            _ = synapse_net.forward_propagation(dummy_spike)
            t_fwd_end = time.perf_counter()
            
            fwd_latency_ms = (t_fwd_end - t_fwd_start) * 1000.0
            print(f"   ↳ Latência de Propagação (W * x): {fwd_latency_ms:.4f} ms")

            del synapse_net
        except Exception as e:
            print(f"{sc['name']:<22} | {hidden_size:<18,d} | {'FALHA / LIMITE DE HARDWARE':<20} | {str(e)}")

    print("="*75)
    print("✅ Benchmark de Escala Concluído com Sucesso!")

if __name__ == "__main__":
    scale_benchmark()