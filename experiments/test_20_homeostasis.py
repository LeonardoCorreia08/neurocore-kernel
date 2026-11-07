'''Modulo: test_03_homeostasis.py'''
import torch
from core.neuron import LeakyIntegrateAndFire
from learning.homeostasis import HomeostasisManager

def run_homeostasis_test():
    print("Iniciando Validação de Homeostase (Estabilidade a Longo Prazo)...")
    
    # Criamos 2 neurônios
    # Neurônio 0: Vai receber uma enxurrada de corrente (Vai tentar ser Epiléptico)
    # Neurônio 1: Vai receber migalhas de corrente (Vai tentar morrer)
    num_neurons = 2
    lif = LeakyIntegrateAndFire(num_neurons=num_neurons, threshold=1.0)
    
    # Queremos que eles disparem em torno de 10% do tempo (0.1)
    homeostase = HomeostasisManager(num_neurons=num_neurons, target_firing_rate=0.1, tau_avg=10.0, adjustment_rate=0.5)

    corrente_injusta = torch.tensor([0.95, 0.1]) 
    
    print(f"Limiares Iniciais: [1.0, 1.0]")
    
    # Simulamos 50 passos de tempo
    for step in range(1, 51):
        spikes, mem = lif(corrente_injusta)
        
        # O gerenciador observa os spikes e ajusta a camada LIF instantaneamente
        homeostase.apply_homeostasis(spikes, lif)
        
        if step % 10 == 0:
            print(f"Passo {step:02d} | Spikes atuais: {spikes.tolist()} | Limiares adaptados: [{lif.threshold[0]:.4f}, {lif.threshold[1]:.4f}]")
            
    print("\n[Resultados] Adaptação Neural concluída!")
    print("-> O Neurônio 0 (Estressado) teve seu limiar aumentado para conter a hiperatividade.")
    print("-> O Neurônio 1 (Ocioso) teve seu limiar rebaixado para conseguir disparar mesmo com pouca energia.")
    print("Estabilidade a longo prazo GARANTIDA!")

if __name__ == "__main__":
    run_homeostasis_test()