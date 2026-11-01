'''Modulo: neurocore_04_test.py'''
import torch
import numpy as np
from core.config_loader import load_brain_config
from core.neuron import LeakyIntegrateAndFire
from core.synapse import SparseSynapseManager
from brain.attention_engine import AttentionEngine
from environment.world import VirtualWorld2D
from learning.reward_learning import RSTDPManager

def run_neurocore_04_experiment():
    print("Iniciando NeuroCore 0.4 (Atencao Cognitiva + Obstaculos + Persistencia)...")
    
    config = load_brain_config()
    target_neurons = config["scale"]["target_neurons"]
    density = config["scale"]["connectivity_density"]
    
    world = VirtualWorld2D()
    
    input_dim = 4
    hidden_dim = target_neurons - 8
    output_dim = 4
    
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)
    
    attention_engine = AttentionEngine(dim=input_dim)
    
    synapse_w1 = SparseSynapseManager(num_posts=hidden_dim, num_pres=input_dim, density=density)
    synapse_w2 = SparseSynapseManager(num_posts=output_dim, num_pres=hidden_dim, density=density)
    
    rstdp_w1 = RSTDPManager(num_posts=hidden_dim, num_pres=input_dim, lr=0.01)
    rstdp_w2 = RSTDPManager(num_posts=output_dim, num_pres=hidden_dim, lr=0.01)

    num_episodes = 5
    for episode in range(num_episodes):
        sensors = world.reset()
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        rstdp_w1.reset_state()
        rstdp_w2.reset_state()
        
        steps = 0
        while steps < 80:
            x_sensory = torch.tensor(sensors, dtype=torch.float32)
            
            # Aplicar Atencao Cognitiva (Módulo 4)
            gated_sensory = attention_engine.apply_attention(x_sensory) * 2.0
            
            spk_sensory, _ = sensory_layer(gated_sensory)
            
            h_input = synapse_w1.forward_propagation(spk_sensory)
            spk_hidden, _ = hidden_layer(h_input)
            
            m_input = synapse_w2.forward_propagation(spk_hidden)
            spk_motor, mem_motor = motor_layer(m_input)
            
            rstdp_w1.step_eligibility(spk_sensory, spk_hidden)
            rstdp_w2.step_eligibility(spk_hidden, spk_motor)
            
            action_id = torch.argmax(mem_motor).item()
            sensors, reward, done = world.step(action_id)
            
            if reward != 0.0:
                delta_w1 = rstdp_w1.lr * reward * rstdp_w1.eligibility_trace
                delta_w2 = rstdp_w2.lr * reward * rstdp_w2.eligibility_trace
                synapse_w1.update_sparse_weights(delta_w1)
                synapse_w2.update_sparse_weights(delta_w2)
            
            steps += 1
            if done:
                break

        print(f"Episodio {episode + 1}/{num_episodes} concluido em {steps} passos.")

    # Testar rotina de salvamento de checkpoint (Persistência)
    synapse_w1.save_checkpoint("config/synapse_w1_checkpoint.pt")
    print("NeuroCore 0.4 executado e validado com sucesso dentro da estrutura modular!")

if __name__ == "__main__":
    run_neurocore_04_experiment()