'''Modulo: test_08_cortical_modules.py'''
import torch
import numpy as np
from core.config_loader import load_brain_config
from core.neuron import LeakyIntegrateAndFire
from core.synapse import SparseSynapseManager
from brain.attention_engine import AttentionEngine
from brain.associative_cortex import AssociativeCortex
from brain.decision_engine import DecisionEngine
from brain.motor_engine import MotorEngine
from environment.world import VirtualWorld2D

def run_cortical_test():
    print("Iniciando Teste Integrado dos Subsistemas Corticais (NeuroCore)...")
    
    config = load_brain_config()
    target_neurons = config["scale"]["target_neurons"]
    density = config["scale"]["connectivity_density"]
    
    world = VirtualWorld2D()
    
    # Dimensoes
    input_dim = 4
    associative_dim = 64
    hidden_dim = target_neurons - 72 # Dividindo o restante dos 10k neurônios
    output_dim = 4
    
    # Instanciar subsistemas corticais
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    attention_engine = AttentionEngine(dim=input_dim)
    associative_cortex = AssociativeCortex(input_dims=[input_dim, output_dim], hidden_dim=associative_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    decision_engine = DecisionEngine(num_choices=output_dim)
    motor_engine = MotorEngine()
    
    synapse_w1 = SparseSynapseManager(num_posts=hidden_dim, num_pres=associative_dim, density=density)
    synapse_w2 = SparseSynapseManager(num_posts=output_dim, num_pres=hidden_dim, density=density)

    print("Rodando ciclo de teste de integracao de 20 passos na arena...")
    sensors = world.reset()
    
    for step in range(20):
        x_sensory = torch.tensor(sensors, dtype=torch.float32)
        
        # 1. Atencao Cognitiva
        gated_sensory = attention_engine.apply_attention(x_sensory)
        spk_sensory, _ = sensory_layer(gated_sensory)
        
        # 2. Cortex Associativo (Integra sensorial com feedback simulado)
        dummy_feedback = torch.zeros(output_dim)
        spk_associative = associative_cortex([spk_sensory, dummy_feedback])
        
        # 3. Propagacao Oculta via Sinapses Esparsas
        h_input = synapse_w1.forward_propagation(spk_associative)
        spk_hidden, _ = hidden_layer(h_input)
        
        # 4. Motor e Decisao Competitiva
        m_input = synapse_w2.forward_propagation(spk_hidden)
        decision_spikes = decision_engine(m_input)
        
        # 5. Decodificacao do Motor
        action_id = motor_engine.decode_action(decision_spikes)
        
        # 6. Interacao com o Ambiente
        sensors, reward, done = world.step(action_id)
        
        print(f"Passo {step + 1}/20 | Acao Escolhida: {action_id} | Recompensa: {reward} | Feito: {done}")
        if done:
            sensors = world.reset()

    print("Teste integrado dos subsistemas corticais concluido com sucesso!")

if __name__ == "__main__":
    run_cortical_test()