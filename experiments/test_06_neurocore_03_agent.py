'''Modulo: test_06_neurocore_03_agent.py'''
import torch
from core.config_loader import load_brain_config
from core.neuron import LeakyIntegrateAndFire
from core.sparse_network import SparseSynapseNetwork
from environment.world import VirtualWorld2D
from learning.reward_learning import RSTDPManager

def run_neurocore_03_agent():
    print("Iniciando NeuroCore 0.3 - Agente Autonomo com 10.000 Neuronios e R-STDP...")
    
    config = load_brain_config()
    # Os parâmetros da arquitetura original deste experimento foram migrados
    # para legacy; a configuração canônica atual mantém rede e neurônio separados.
    target_neurons = config["legacy"]["scale"]["target_neurons"]
    density = config["legacy"]["scale"]["connectivity_density"]
    threshold = config["neuron"]["threshold"]
    decay = config["neuron"]["decay"]
    
    world = VirtualWorld2D(config=config)
    
    # Camadas neurais na escala de grande porte
    # Para adaptar 10k neurônios ao espaço de 4 entradas e 4 saídas, 
    # dividimos o núcleo: 4 sensores -> 9992 neurônios ocultos -> 4 motores
    input_dim = 4
    hidden_dim = target_neurons - 8  # ~9992 neurônios
    output_dim = 4
    
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim, threshold=threshold, decay=decay, config=config)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim, threshold=threshold, decay=decay, config=config)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim, threshold=threshold, decay=decay, config=config)
    
    print("Construindo redes de sinapses esparsas (W1: Sensorial -> Oculta, W2: Oculta -> Motora)...")
    # W1: matriz esparsa [hidden_dim x input_dim]
    synapse_w1 = SparseSynapseNetwork(num_posts=hidden_dim, num_pres=input_dim, density=density, weight_scale=0.2)
    # W2: matriz esparsa [output_dim x hidden_dim]
    synapse_w2 = SparseSynapseNetwork(num_posts=output_dim, num_pres=hidden_dim, density=density, weight_scale=0.2)
    
    # Gerenciadores de R-STDP para as redes esparsas
    rstdp_w1 = RSTDPManager(num_posts=hidden_dim, num_pres=input_dim, lr=0.01)
    rstdp_w2 = RSTDPManager(num_posts=output_dim, num_pres=hidden_dim, lr=0.01)

    num_episodes = 10  # Menos episódios devido à alta densidade de cálculo de 10k neurônios
    total_goals_reached = 0

    for episode in range(num_episodes):
        sensors = world.reset()
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        rstdp_w1.reset_state()
        rstdp_w2.reset_state()
        
        steps = 0
        episode_reward = 0.0
        
        while steps < 100: # Limitado a 100 passos por episódio para agilizar o teste em 10k
            x_sensory = torch.tensor(sensors, dtype=torch.float32) * 2.0
            
            # 1. Propagacao Sensorial
            spk_sensory, _ = sensory_layer(x_sensory)
            
            # 2. Propagacao Oculta com Matriz Esparsa (W1 * spk_sensory)
            h_input = synapse_w1.forward_propagation(spk_sensory)
            spk_hidden, _ = hidden_layer(h_input)
            
            # 3. Propagacao Motora com Matriz Esparsa (W2 * spk_hidden)
            m_input = synapse_w2.forward_propagation(spk_hidden)
            spk_motor, mem_motor = motor_layer(m_input)
            
            # 4. Rastreamento de Elegibilidade R-STDP
            indices_w1 = synapse_w1.sparse_weights.coalesce().indices()
            indices_w2 = synapse_w2.sparse_weights.coalesce().indices()
            rstdp_w1.step_eligibility(spk_sensory, spk_hidden, indices_w1)
            rstdp_w2.step_eligibility(spk_hidden, spk_motor, indices_w2)
            
            # 5. Decisao de Acao
            action_id = torch.argmax(mem_motor).item()
            sensors, reward, done = world.step(action_id)
            
            # 6. Modulacao por Recompensa (Dopamina Global)
            if reward > 0:
                # Atualizar valores na matriz esparsa com base na elegibilidade modulada
                delta_w1 = rstdp_w1.apply_reward_modulation(reward, indices_w1)
                delta_w2 = rstdp_w2.apply_reward_modulation(reward, indices_w2)

                synapse_w1.update_sparse_weights(delta_w1)
                synapse_w2.update_sparse_weights(delta_w2)
                
                total_goals_reached += 1
            
            episode_reward += reward
            steps += 1
            if done:
                break

        print(f"Episodio {episode + 1}/{num_episodes} | Passos: {steps} | Metas Atingidas: {total_goals_reached}")

    print("Experimento NeuroCore 0.3 (Escala 10k + R-STDP + Ambiente 2D) concluido com sucesso!")

if __name__ == "__main__":
    run_neurocore_03_agent()
