'''Modulo: learning_test.py'''
import torch
import numpy as np
from core.neuron import LeakyIntegrateAndFire
from environment.world import VirtualWorld2D
from learning.reward_learning import RSTDPManager
from memory.short_term import ShortTermMemory
from monitoring.spike_visualizer import SpikeVisualizer

def run_neurocore_02_experiment():
    print("Iniciando NeuroCore 0.2 - Com Memory Engine e Raster Plot...")
    
    world = VirtualWorld2D()
    
    input_dim = 4
    hidden_dim = 16
    output_dim = 4
    
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    # Módulo 2: Memoria de curto prazo
    memory_engine = ShortTermMemory(hidden_dim=hidden_dim, history_steps=5)
    
    W1 = torch.randn(hidden_dim, input_dim) * 0.5
    W2 = torch.randn(output_dim, hidden_dim) * 0.5
    W_mem = torch.randn(hidden_dim, hidden_dim) * 0.2  # Conexoes recorrentes de memoria

    # A API R-STDP recebe coordenadas [post, pre] das conexoes existentes.
    # Como W1/W2 sao densas, todas as conexoes sao representadas.
    def dense_sparse_indices(num_posts, num_pres):
        post_idx, pre_idx = torch.meshgrid(
            torch.arange(num_posts), torch.arange(num_pres), indexing="ij"
        )
        return torch.stack((post_idx.reshape(-1), pre_idx.reshape(-1)))

    indices_w1 = dense_sparse_indices(hidden_dim, input_dim)
    indices_w2 = dense_sparse_indices(output_dim, hidden_dim)

    rstdp_w1 = RSTDPManager(num_posts=hidden_dim, num_pres=input_dim, lr=0.05)
    rstdp_w2 = RSTDPManager(num_posts=output_dim, num_pres=hidden_dim, lr=0.05)

    num_episodes = 30
    total_goals_reached = 0
    last_episode_spikes = []  # Para gerar o raster plot do ultimo episodio

    for episode in range(num_episodes):
        sensors = world.reset()
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        memory_engine.reset()
        rstdp_w1.reset_state()
        rstdp_w2.reset_state()
        
        episode_spikes = []
        steps = 0
        
        while steps < 200:
            x_sensory = torch.tensor(sensors, dtype=torch.float32)
            
            spk_sensory, _ = sensory_layer(x_sensory)
            
            # Entrada oculta = Sensores + Contexto da Memoria de Curto Prazo
            memory_context = memory_engine.get_context()
            h_input = torch.matmul(W1, spk_sensory) + torch.matmul(W_mem, memory_context)
            
            spk_hidden, _ = hidden_layer(h_input)
            
            # Atualizar memoria com os spikes atuais
            memory_engine.update(spk_hidden)
            
            if episode == num_episodes - 1:
                episode_spikes.append(spk_hidden)
            
            m_input = torch.matmul(W2, spk_hidden)
            spk_motor, mem_motor = motor_layer(m_input)
            
            rstdp_w1.step_eligibility(spk_sensory, spk_hidden, indices_w1)
            rstdp_w2.step_eligibility(spk_hidden, spk_motor, indices_w2)
            
            action_id = torch.argmax(mem_motor).item()
            sensors, reward, done = world.step(action_id)
            
            if reward > 0:
                delta_w1 = rstdp_w1.apply_reward_modulation(reward, indices_w1)
                delta_w2 = rstdp_w2.apply_reward_modulation(reward, indices_w2)
                W1[indices_w1[0], indices_w1[1]] += delta_w1
                W2[indices_w2[0], indices_w2[1]] += delta_w2
                total_goals_reached += 1
            
            steps += 1
            if done:
                break

        print(f"Episodio {episode + 1}/{num_episodes} | Passos: {steps} | Metas Atingidas: {total_goals_reached}")

    # Salvar Raster Plot do ultimo episodio executado
    SpikeVisualizer.plot_raster(episode_spikes, filename="docs/spike_raster.png")
    print("Experimento NeuroCore 0.2 concluido com sucesso!")

if __name__ == "__main__":
    run_neurocore_02_experiment()
