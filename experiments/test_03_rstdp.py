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

    # A API R-STDP recebe coordenadas [post, pre] das sinapses existentes.
    # Estas matrizes são densas, portanto todas as conexões são representadas.
    def dense_sparse_indices(num_posts, num_pres):
        post_idx, pre_idx = torch.meshgrid(
            torch.arange(num_posts), torch.arange(num_pres), indexing="ij"
        )
        return torch.stack((post_idx.reshape(-1), pre_idx.reshape(-1)))

    indices_w1 = dense_sparse_indices(hidden_dim, input_dim)
    indices_w2 = dense_sparse_indices(output_dim, hidden_dim)
    
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
            rstdp_w1.step_eligibility(spk_sensory, spk_hidden, indices_w1)
            rstdp_w2.step_eligibility(spk_hidden, spk_motor, indices_w2)
            
            # Decisao de acao baseada no potencial motor
            action_id = torch.argmax(mem_motor).item()
            
            # Executar acao no ambiente
            sensors, reward, done = world.step(action_id)
            
            # 2. Se o agente encontrou a meta, disparar o sinal global de recompensa (Dopamina)
            if reward > 0:
                delta_w1 = rstdp_w1.apply_reward_modulation(reward, indices_w1)
                delta_w2 = rstdp_w2.apply_reward_modulation(reward, indices_w2)
                W1[indices_w1[0], indices_w1[1]] += delta_w1
                W2[indices_w2[0], indices_w2[1]] += delta_w2
                total_goals_reached += 1
            
            episode_reward += reward
            steps += 1
            
            if done:
                break

        print(f"Episodio {episode + 1}/{num_episodes} | Passos: {steps} | Metas Atingidas: {total_goals_reached}")

    print("Experimento NeuroCore 0.1 com R-STDP concluido com sucesso!")

if __name__ == "__main__":
    run_neurocore_01_experiment()
