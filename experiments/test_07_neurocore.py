'''Modulo: neurocore_04_test.py'''
from pathlib import Path

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
    # A escala original deste experimento permanece explicitamente em legacy;
    # os parâmetros de rede/neuronais canônicos são lidos pelas próprias APIs.
    target_neurons = config["legacy"]["scale"]["target_neurons"]
    density = config["legacy"]["scale"]["connectivity_density"]

    world = VirtualWorld2D(config=config)

    input_dim = 4
    hidden_dim = target_neurons - 8
    output_dim = 4

    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim, config=config)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim, config=config)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim, config=config)

    attention_engine = AttentionEngine(dim=input_dim, config=config)

    synapse_w1 = SparseSynapseManager(
        num_posts=hidden_dim, num_pres=input_dim, density=density, config=config
    )
    synapse_w2 = SparseSynapseManager(
        num_posts=output_dim, num_pres=hidden_dim, density=density, config=config
    )

    rstdp_w1 = RSTDPManager(
        num_posts=hidden_dim, num_pres=input_dim, config=config
    )
    rstdp_w2 = RSTDPManager(
        num_posts=output_dim, num_pres=hidden_dim, config=config
    )

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

            # Aplicar Atenção Cognitiva (Módulo 4).
            gated_sensory = attention_engine.apply_attention(x_sensory) * 2.0

            spk_sensory, _ = sensory_layer(gated_sensory)

            h_input = synapse_w1.forward_propagation(spk_sensory)
            spk_hidden, _ = hidden_layer(h_input)

            m_input = synapse_w2.forward_propagation(spk_hidden)
            spk_motor, mem_motor = motor_layer(m_input)

            indices_w1 = synapse_w1.sparse_weights.coalesce().indices()
            indices_w2 = synapse_w2.sparse_weights.coalesce().indices()
            rstdp_w1.step_eligibility(spk_sensory, spk_hidden, indices_w1)
            rstdp_w2.step_eligibility(spk_hidden, spk_motor, indices_w2)

            action_id = torch.argmax(mem_motor).item()
            sensors, reward, done = world.step(action_id)

            if reward != 0.0:
                delta_w1 = rstdp_w1.apply_reward_modulation(reward, indices_w1)
                delta_w2 = rstdp_w2.apply_reward_modulation(reward, indices_w2)
                synapse_w1.update_sparse_weights(delta_w1)
                synapse_w2.update_sparse_weights(delta_w2)

            steps += 1
            if done:
                break

        print(f"Episodio {episode + 1}/{num_episodes} concluido em {steps} passos.")

    # Persistir topologia e pesos sem depender de método ausente no manager legado.
    checkpoint_path = Path(config["_config_path"]).parent / "synapse_w1_checkpoint.pt"
    weights = synapse_w1.sparse_weights.coalesce()
    torch.save(
        {
            "indices": weights.indices().cpu(),
            "values": weights.values().cpu(),
            "size": tuple(weights.size()),
            "density": synapse_w1.density,
            "weight_min": synapse_w1.weight_min,
            "weight_max": synapse_w1.weight_max,
            "weight_init_scale": synapse_w1.weight_init_scale,
        },
        checkpoint_path,
    )
    print(f"Checkpoint salvo em '{checkpoint_path}'.")
    print("NeuroCore 0.4 executado e validado com sucesso dentro da estrutura modular!")


if __name__ == "__main__":
    run_neurocore_04_experiment()
