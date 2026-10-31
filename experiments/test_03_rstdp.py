'''Modulo: learning_test.py'''
import torch
import numpy as np
from core.neuron import LeakyIntegrateAndFire
from environment.world import VirtualWorld2D
from learning.reward_learning import RSTDPManager

def run_neurocore_01_experiment():
    print("Iniciando NeuroCore 0.1 - Experimento com R-STDP (Modulado por Recompensa)...")
    
    world = VirtualWorld2D()
    
    input_dim = 4
    hidden_dim = 16
    output_dim = 4
    
    # Camadas LIF
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    # Pesos sinapticos iniciais
    W1 = torch.randn(hidden_dim, input_dim) * 0.1
    W2 = torch.randn(output_dim, hidden_dim) * 0.1
    
    # Instanciando R-STDP para W1 e W2
    rstdp_w1 = RSTDPManager(num_posts=hidden_dim, num_pres=input_dim, lr=0.05)
    rstdp_w2 = RSTDPManager(num_posts=output_dim, num_pres=hidden_dim, lr=0.05)

    num_episodes = 50
    total_goals_reached = 0

    for episode in range(num_episodes):
        sensors = world.reset()
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        rstdp_w1.reset_state()
        rstdp_w2.reset_state()
        
        episode_reward = 0.0
        steps = 0
        
        while steps < 200:
            x_sensory = torch.tensor(sensors, dtype=torch.float32)
            
            # Propagacao neural
            spk_sensory, _ = sensory_layer(x_sensory)
            
            h_input = torch.matmul(W1, spk_sensory)
            spk_hidden, _ = hidden_layer(h_input)
            
            m_input = torch.matmul(W2, spk_hidden)
            spk_motor, mem_motor = motor_layer(m_input)
            
            # 1. Acumular elegibilidade sinaptica a cada passo (sem mudar peso ainda)
            rstdp_w1.step_eligibility(spk_sensory, spk_hidden)
            rstdp_w2.step_eligibility(spk_hidden, spk_motor)
            
            # Decisao de acao baseada no potencial motor
            action_id = torch.argmax(mem_motor).item()
            
            # Executar acao no ambiente
            sensors, reward, done = world.step(action_id)
            
            # 2. Se o agente encontrou a meta, disparar o sinal global de recompensa (Dopamina)
            if reward > 0:
                W1 = rstdp_w1.apply_reward_modulation(W1, reward)
                W2 = rstdp_w2.apply_reward_modulation(W2, reward)
                total_goals_reached += 1
            
            episode_reward += reward
            steps += 1
            
            if done:
                break

        print(f"Episodio {episode + 1}/{num_episodes} | Passos: {steps} | Metas Atingidas: {total_goals_reached}")

    print("Experimento NeuroCore 0.1 com R-STDP concluido com sucesso!")

if __name__ == "__main__":
    run_neurocore_01_experiment()