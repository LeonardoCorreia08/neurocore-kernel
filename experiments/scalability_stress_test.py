'''Modulo: experiments/scalability_stress_test.py'''
import time
import torch
from core.spike_engine import SpikeEngine

def run_scalability_stress_test():
    print("="*65)
    print("🔥 ITEM 27: STRESS TEST DE ESCALABILIDADE (DENSIDADE & TAMANHO)")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Matriz de cenários de teste: [Tamanho da Camada Oculta, Densidade Sináptica]
    scenarios = [
        {"hidden": 512,  "density": 0.05, "name": "Leve & Muito Esparsa"},
        {"hidden": 1024, "density": 0.15, "name": "Média (Padrão)"},
        {"hidden": 4096, "density": 0.10, "name": "Grande & Esparsa"},
        {"hidden": 8192, "density": 0.20, "name": "Massiva (Stress Extremo)"}
    ]

    print(f"{'Cenário':<25} | {'Neurónios':<10} | {'Densidade':<10} | {'Latência (ms/tick)':<20}")
    print("-" * 75)

    for sc in scenarios:
        hidden_size = sc["hidden"]
        density = sc["density"]
        layer_sizes = [16, hidden_size, 4]
        
        try:
            # Inicializa a rede para o cenário
            engine = SpikeEngine(layer_sizes=layer_sizes, density=density, device=str(device))
            dummy_input = torch.randn(16, device=device)
            
            # Aquecimento rápido
            for _ in range(5):
                _ = engine(dummy_input)
                
            # Medição de 100 passos
            steps = 100
            start = time.perf_counter()
            for _ in range(steps):
                _ = engine(dummy_input)
            end = time.perf_counter()
            
            latency_ms = ((end - start) / steps) * 1000.0
            print(f"{sc['name']:<25} | {hidden_size:<10} | {density*100:->8.1f}% | {latency_ms:<20.2f}")
            
            # Limpeza de memória
            del engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"{sc['name']:<25} | {hidden_size:<10} | {density*100:->8.1f}% | FALHA / OOM")

    print("="*65)
    print("✅ Stress Test de Escalabilidade Concluído com Sucesso!")

def run_multi_layer_test():
    print("\n" + "="*65)
    print("🔗 TESTE EXTRA: PROFUNDIDADE DA REDE (Multi-Layer Deep SNN)")
    print("="*65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Rede profunda com 5 camadas: [16, 512, 512, 512, 4]
    deep_layers = [16, 512, 512, 512, 4]
    print(f"Instanciando Deep SNN com arquitetura: {deep_layers}")
    
    engine = SpikeEngine(layer_sizes=deep_layers, density=0.10, device=str(device))
    dummy_input = torch.randn(16, device=device)
    
    start = time.perf_counter()
    for _ in range(100):
        _ = engine(dummy_input)
    end = time.perf_counter()
    
    deep_latency = ((end - start) / 100.0) * 1000.0
    print(f"Latência Média da Deep SNN (100 ticks): {deep_latency:.2f} ms / tick")
    print("="*65)

if __name__ == "__main__":
    run_scalability_stress_test()
    run_multi_layer_test()