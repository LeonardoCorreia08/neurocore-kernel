'''Modulo: learning_test.py (NeuroCore 0.2 Otimizado)'''
import torch
import numpy as np
from core.neuron import LeakyIntegrateAndFire
from environment.world import VirtualWorld2D
from learning.reward_learning import RSTDPManager
from memory.short_term import ShortTermMemory
from monitoring.spike_visualizer import SpikeVisualizer


def apply_synaptic_homeostasis(W, target_row_mean, w_min=-1.0, w_max=1.0):
    W_clamped = torch.clamp(W, min=w_min, max=w_max)
    current_row_mean = W_clamped.mean(dim=1, keepdim=True)
    return torch.clamp(W_clamped - current_row_mean + target_row_mean, min=w_min, max=w_max)


def select_motor_action(mem_motor: torch.Tensor, epsilon: float = 0.1) -> int:
    """Seleção de ação com exploração Epsilon-Greedy."""
    if np.random.random() < epsilon:
        return int(np.random.randint(0, mem_motor.numel()))
    return int(torch.argmax(mem_motor).item())


def run_neurocore_02_experiment():
    print("Iniciando NeuroCore 0.2 Otimizado - Com Memory Engine, Exploração e Raster Plot...")
    
    torch.manual_seed(42)
    np.random.seed(42)

    world = VirtualWorld2D()
    
    input_dim = 4
    hidden_dim = 16
    output_dim = 4
    snn_ticks_per_step = 5
    
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    # Memória de curto prazo
    memory_engine = ShortTermMemory(hidden_dim=hidden_dim, history_steps=5)
    
    # Inicialização fan-in saudável
    W1 = torch.randn(hidden_dim, input_dim) * (1.0 / np.sqrt(input_dim))
    W1 = apply_synaptic_homeostasis(W1, target_row_mean=0.12)
    
    W2 = torch.randn(output_dim, hidden_dim) * (1.0 / np.sqrt(hidden_dim))
    W2 = apply_synaptic_homeostasis(W2, target_row_mean=0.06)
    
    W_mem = torch.randn(hidden_dim, hidden_dim) * 0.1
    W_mem.fill_diagonal_(0.0)

    def dense_sparse_indices(num_posts, num_pres):
        post_idx, pre_idx = torch.meshgrid(
            torch.arange(num_posts), torch.arange(num_pres), indexing="ij"
        )
        return torch.stack((post_idx.reshape(-1), pre_idx.reshape(-1)))

    indices_w1 = dense_sparse_indices(hidden_dim, input_dim)
    indices_w2 = dense_sparse_indices(output_dim, hidden_dim)

    rstdp_w1 = RSTDPManager(num_posts=hidden_dim, num_pres=input_dim, lr=0.015)
    rstdp_w2 = RSTDPManager(num_posts=output_dim, num_pres=hidden_dim, lr=0.008)

    num_episodes = 30
    total_goals_reached = 0
    episode_spikes = []

    for episode in range(num_episodes):
        sensors = world.reset()
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        memory_engine.reset()
        rstdp_w1.reset_state()
        rstdp_w2.reset_state()
        
        current_episode_spikes = []
        steps = 0
        epsilon = max(0.05, 0.30 * (1.0 - episode / num_episodes))
        
        while steps < 200:
            x_sensory = torch.tensor(sensors, dtype=torch.float32)
            motor_spike_counts = torch.zeros(output_dim)
            hidden_spike_accum = torch.zeros(hidden_dim)
            
            # Dinâmica SNN com micro-ticks
            for _ in range(snn_ticks_per_step):
                sensory_drive = x_sensory + (torch.randn(input_dim) * 0.05)
                spk_sensory, _ = sensory_layer(sensory_drive)
                
                memory_context = memory_engine.get_context()
                h_input = torch.matmul(W1, spk_sensory) + torch.matmul(W_mem, memory_context) + (torch.randn(hidden_dim) * 0.1)
                spk_hidden, _ = hidden_layer(h_input)
                
                m_input = torch.matmul(W2, spk_hidden) + (torch.randn(output_dim) * 0.1)
                spk_motor, mem_motor = motor_layer(m_input)
                
                hidden_layer.mem.clamp_(min=-0.5)
                motor_layer.mem.clamp_(min=-0.5)
                
                rstdp_w1.step_eligibility(spk_sensory, spk_hidden, indices_w1)
                rstdp_w2.step_eligibility(spk_hidden, spk_motor, indices_w2)
                
                motor_spike_counts += spk_motor
                hidden_spike_accum += spk_hidden

            # Atualiza memória de curto prazo com os spikes acumulados no passo
            active_hidden = (hidden_spike_accum > 0).float()
            memory_engine.update(active_hidden)
            
            if episode == num_episodes - 1:
                current_episode_spikes.append(active_hidden)

            action_id = select_motor_action(mem_motor, epsilon)
            sensors, reward, done = world.step(action_id)
            steps += 1
            
            # Modulação por reforço (Sucesso e Colisão)
            dopamine_signal = 0.0
            if reward > 0.1:
                dopamine_signal = 2.0
                total_goals_reached += 1
            elif reward < -0.1:
                dopamine_signal = -0.3
                
            if dopamine_signal != 0.0:
                delta_w1 = rstdp_w1.apply_reward_modulation(dopamine_signal, indices_w1)
                delta_w2 = rstdp_w2.apply_reward_modulation(dopamine_signal, indices_w2)
                
                W1[indices_w1[0], indices_w1[1]] += delta_w1
                W2[indices_w2[0], indices_w2[1]] += delta_w2
                
                W1 = apply_synaptic_homeostasis(W1, target_row_mean=0.12)
                W2 = apply_synaptic_homeostasis(W2, target_row_mean=0.06)

            if done:
                break

        if episode == num_episodes - 1:
            episode_spikes = current_episode_spikes

        print(f"Episodio {episode + 1}/{num_episodes} | Passos: {steps:03d} | Metas Acumuladas: {total_goals_reached}")

    # Salvar Raster Plot do último episódio
    SpikeVisualizer.plot_raster(episode_spikes, filename="docs/spike_raster.png")
    print("Experimento NeuroCore 0.2 otimizado concluído com sucesso!")

if __name__ == "__main__":
    run_neurocore_02_experiment()