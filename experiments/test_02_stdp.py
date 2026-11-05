'''Modulo: learning_test.py'''
import torch
from core.neuron import LeakyIntegrateAndFire
from environment.world import VirtualWorld2D
from learning.stdp import STDPManager


def run_neurocore_01_experiment():
    print("Iniciando NeuroCore 0.1 - Experimento com STDP Ativo...")
    
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
    
    # Instanciando gerenciadores STDP para W1 e W2
    stdp_w1 = STDPManager(num_posts=hidden_dim, num_pres=input_dim, lr=0.01)
    stdp_w2 = STDPManager(num_posts=output_dim, num_pres=hidden_dim, lr=0.01)
    # A API atual recebe índices (pós, pré); nesta rede pequena, todas as
    # conexões são representadas para manter a atualização densa original.
    indices_w1 = torch.cartesian_prod(torch.arange(hidden_dim), torch.arange(input_dim)).T
    indices_w2 = torch.cartesian_prod(torch.arange(output_dim), torch.arange(hidden_dim)).T

    num_episodes = 50
    total_goals_reached = 0

    for episode in range(num_episodes):
        sensors = world.reset()
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        stdp_w1.reset_traces()
        stdp_w2.reset_traces()
        
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
            
            # Atualizacao STDP dos pesos em tempo real com base nos spikes
            delta_w1 = stdp_w1.compute_sparse_deltas(spk_sensory, spk_hidden, indices_w1)
            delta_w2 = stdp_w2.compute_sparse_deltas(spk_hidden, spk_motor, indices_w2)
            W1 = W1 + delta_w1.view_as(W1)
            W2 = W2 + delta_w2.view_as(W2)
            
            # Decisao de acao
            action_id = torch.argmax(mem_motor).item()
            
            sensors, reward, done = world.step(action_id)
            episode_reward += reward
            steps += 1
            
            if done:
                if reward > 0:
                    total_goals_reached += 1
                break

        print(f"Episodio {episode + 1}/{num_episodes} | Passos: {steps} | Metas Atingidas: {total_goals_reached}")

    print("Experimento NeuroCore 0.1 com STDP concluido com sucesso!")

if __name__ == "__main__":
    run_neurocore_01_experiment()
