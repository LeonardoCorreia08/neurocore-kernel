'''Modulo: test_01_lif_base.py'''
import torch
import numpy as np
from core.neuron import LeakyIntegrateAndFire
from environment.world import VirtualWorld2D

# =====================================================================
# 1. TESTE UNITÁRIO: FÍSICA DA MEMBRANA LIF E GERENCIAMENTO DE ESTADO
# =====================================================================
def run_lif_membrane_test():
    print("Iniciando Teste Dinâmico de Membrana (LIF)...")
    
    # Inicializa 1 neurônio com limiar de 1.0 e retenção de 0.9 (90%)
    lif = LeakyIntegrateAndFire(num_neurons=1, threshold=1.0, decay=0.9)
    
    print("\n--- Fase 1: Acumulação e Disparo ---")
    corrente = torch.tensor([0.4])
    
    for step in range(1, 6):
        spike, mem = lif(corrente)
        estado = "🔥 DISPARO e Reset!" if spike.item() > 0 else "Acumulando..."
        print(f"Passo {step} | Corrente: 0.4 | Voltagem: {mem.item():.4f} | {estado}")

    print("\n--- Fase 2: Decaimento (Leak) ---")
    sem_corrente = torch.tensor([0.0])
    for step in range(6, 9):
        spike, mem = lif(sem_corrente)
        print(f"Passo {step} | Corrente: 0.0 | Voltagem: {mem.item():.4f} | Vazamento")

    print("\n--- Fase 3: Reset Manual ---")
    lif.reset_state()
    print(f"Após lif.reset_state() | Voltagem: {lif.mem.item():.4f}\n")


# =====================================================================
# 2. EXPERIMENTO PRINCIPAL: APRENDIZADO NO AMBIENTE VIRTUAL
# =====================================================================
def run_neurocore_01_experiment():
    print("Iniciando NeuroCore 0.1 - Experimento de Aprendizado Autonomo...")
    
    world = VirtualWorld2D()
    
    # Camada neural: 4 entradas sensoriais convertidas, 16 neuronios internos, 4 saidas (acoes)
    input_dim = 4
    hidden_dim = 16
    output_dim = 4
    
    # Instanciando camadas LIF
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    # Pesos sinapticos aleatorios iniciais (simulando a conectividade sinaptica)
    W1 = torch.randn(hidden_dim, input_dim) * 0.1
    W2 = torch.randn(output_dim, hidden_dim) * 0.1

    num_episodes = 50
    total_goals_reached = 0

    for episode in range(num_episodes):
        sensors = world.reset()
        
        # PROVA DA MELHORIA 6: O reset_state() agora funciona perfeitamente, 
        # impedindo que a energia de um episódio vaze para o outro!
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        
        episode_reward = 0.0
        steps = 0
        
        while steps < 200:
            # 1. Converter sensores em corrente de entrada para a rede
            x_sensory = torch.tensor(sensors, dtype=torch.float32)
            
            # 2. Propagacao de impulsos (Spikes) atraves da rede
            spk_sensory, _ = sensory_layer(x_sensory)
            
            # Camada oculta recebe corrente ponderada dos sensores
            h_input = torch.matmul(W1, spk_sensory)
            spk_hidden, _ = hidden_layer(h_input)
            
            # Camada motora recebe corrente da camada oculta
            m_input = torch.matmul(W2, spk_hidden)
            spk_motor, mem_motor = motor_layer(m_input)
            
            # 3. Decisao de acao com base no neuronio motor que disparou mais forte
            action_id = torch.argmax(mem_motor).item()
            
            # 4. Executar acao no ambiente virtual
            sensors, reward, done = world.step(action_id)
            episode_reward += reward
            steps += 1
            
            if done:
                total_goals_reached += get_goal_counter(reward)
                break # Quebra o loop se o episódio terminou, otimizando o processamento

        print(f"Episodio {episode + 1}/{num_episodes} | Passos: {steps} | Metas Atingidas Acumuladas: {total_goals_reached}")

    print("Experimento NeuroCore 0.1 concluido com sucesso!")

def get_goal_counter(reward):
    return 1 if reward > 0 else 0

# =====================================================================
# EXECUÇÃO SEQUENCIAL
# =====================================================================
if __name__ == "__main__":
    # Primeiro executa o teste isolado da membrana
    run_lif_membrane_test()
    
    print("=" * 60)
    
    # Depois executa o seu teste original do mundo virtual
    run_neurocore_01_experiment()