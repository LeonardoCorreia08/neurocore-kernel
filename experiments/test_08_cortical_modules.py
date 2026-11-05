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
    torch.manual_seed(42)
    
    config = load_brain_config()
    # A escala desse experimento foi preservada no namespace de migração.
    target_neurons = config["legacy"]["scale"]["target_neurons"]
    density = config["legacy"]["scale"]["connectivity_density"]
    
    world = VirtualWorld2D(config=config)
    
    # Dimensoes
    input_dim = 4
    associative_dim = 64
    hidden_dim = target_neurons - 72 # Dividindo o restante dos 10k neurônios
    output_dim = 4
    
    # Instanciar subsistemas corticais
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim, config=config)
    attention_engine = AttentionEngine(dim=input_dim, config=config)
    associative_cortex = AssociativeCortex(
        input_dims=[input_dim, output_dim], hidden_dim=associative_dim, config=config
    )
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim, config=config)
    decision_engine = DecisionEngine(num_choices=output_dim, config=config)
    motor_engine = MotorEngine(config=config)
    
    synapse_w1 = SparseSynapseManager(
        num_posts=hidden_dim, num_pres=associative_dim, density=density, config=config
    )
    synapse_w2 = SparseSynapseManager(
        num_posts=output_dim, num_pres=hidden_dim, density=density, config=config
    )

    print("Rodando ciclo de teste de integracao de 20 passos na arena...")
    sensors = world.reset()
    action_counts = [0] * output_dim
    total_decision_spikes = 0
    
    for step in range(20):
        x_sensory = torch.tensor(sensors, dtype=torch.float32)
        
        # 1. Atencao Cognitiva
        # Compensa a atenuação da atenção para que o sinal possa alcançar o limiar LIF.
        gated_sensory = attention_engine.apply_attention(x_sensory) * 2.0
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
        action_counts[action_id] += 1
        total_decision_spikes += int(decision_spikes.sum().item())
        
        # 6. Interacao com o Ambiente
        sensors, reward, done = world.step(action_id)
        
        print(f"Passo {step + 1}/20 | Acao Escolhida: {action_id} | Recompensa: {reward} | Feito: {done}")
        if done:
            sensors = world.reset()

    print(f"[Diagnóstico] Spikes de decisão: {total_decision_spikes}; ações: {action_counts}")
    if total_decision_spikes == 0:
        print("[AVISO] Nenhum neurônio de decisão disparou; ação 0 pode ser apenas o fallback sem spikes.")
    elif sum(count > 0 for count in action_counts) == 1:
        print("[AVISO] Uma única ação foi selecionada; a integração não demonstrou diversidade comportamental.")
    else:
        print("[PASS] O circuito produziu spikes de decisão e mais de uma ação.")
    print("Ciclo de integração concluído; este teste não mede aprendizagem nem sucesso em metas.")

if __name__ == "__main__":
    run_cortical_test()
