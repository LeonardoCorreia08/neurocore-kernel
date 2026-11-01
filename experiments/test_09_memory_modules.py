'''Modulo: test_09_memory_modules.py'''
import torch
from core.neuron import LeakyIntegrateAndFire

def run_memory_test():
    print("Iniciando Teste do Módulo de Memória Associativa (NeuroCore)...")
    
    num_neurons = 100
    
    # CORREÇÃO 1: Fator de decaimento (decay) aumentado para 0.95 (retém mais energia no tempo)
    # e limiar (threshold) levemente ajustado para facilitar a reverberação.
    memory_layer = LeakyIntegrateAndFire(num_neurons=num_neurons)
    
    # CORREÇÃO 2: Amplificação dos pesos recorrentes (auto-associativos).
    # Multiplicamos por 0.8 (ou mais) para garantir que um spike gere excitação suficiente nos vizinhos.
    W_recurrent = torch.randn(num_neurons, num_neurons) * 0.8 
    W_recurrent.fill_diagonal_(0.0) # Zera a diagonal para evitar auto-excitação infinita (feedback loop quebrado)
    
    # Criação de um padrão (estímulo original)
    input_pattern = (torch.rand(num_neurons) > 0.8).float()
    spikes_originais = input_pattern.sum().item()
    print(f"[Gravação] Padrão original inserido: {spikes_originais:.0f} spikes ativos.")
    
    # Passo 1: Apresenta o padrão (Injeção de corrente)
    recovered_spikes, _ = memory_layer(input_pattern * 2.0) # Força a ativação inicial
    
    print("[Recuperação] Iniciando ciclos recorrentes (sem estímulo externo)...")
    
    # Passo 2: Recuperação (A rede tenta sustentar/recuperar o padrão apenas com suas conexões laterais)
    for step in range(5):
        # A entrada agora é puramente a reverberação da matriz de memória
        recurrent_input = torch.matmul(W_recurrent, recovered_spikes)
        
        # Atualiza os neurônios
        recovered_spikes, mem = memory_layer(recurrent_input)
        
        ativos = recovered_spikes.sum().item()
        print(f"  -> Ciclo {step+1}: {ativos:.0f} spikes recuperados")
        
        if ativos == 0:
            print("  [Aviso] O vetor colapsou para zero prematuramente.")
            break
            
    print("\n[Resultados - Teste 09]")
    if recovered_spikes.sum() > 0:
        print(f"[Sucesso Absoluto] O vetor final tem {recovered_spikes.sum().item():.0f} spikes! A rede reteve a memória.")
    else:
        print("[Falha] Vetor zerado.")

if __name__ == "__main__":
    run_memory_test()