"""Teste integrado de episódios, replay, consolidação e restauração do NeuroCore."""

import random

import numpy as np
import torch

from core.simulation import TemporalSimulator
from core.spike_engine import SpikeEngine
from environment.world import VirtualWorld2D
from learning.continual_learning import ContinualLearningAgent
from memory.manager import MemoryManager
from monitoring.telemetry import TelemetryManager
from perception.sensor_encoder import SensorEncoder


def run_phase3_full_test():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    print("INICIANDO TESTE INTEGRADO DA MEMÓRIA NEUROCORE\n")
    world = VirtualWorld2D(width=100, height=100)
    engine = SpikeEngine(layer_sizes=[4, 32, 4], density=0.65, weight_scale=0.5)
    simulator = TemporalSimulator(engine, time_steps=10)
    memory = MemoryManager(engine, save_dir="checkpoints_test/", max_buffer_size=1000)
    encoder = SensorEncoder(max_rate=100.0)
    agent = ContinualLearningAgent(simulator, memory, encoder, use_td_critic=True)
    telemetry = TelemetryManager()

    print("[FASE A] Coletando transições state/action/reward/next_state")
    state = world.reset()
    for _ in range(20):
        action = agent.step(state)
        next_state, reward, done = world.step(action)
        agent.process_feedback(
            state, action, reward,
            next_state=next_state,
            done=done,
            task_id="world_2d",
        )
        telemetry.log_step(state, action, reward)
        state = next_state
        if done:
            break
    telemetry.print_summary("Episódio 1")

    # Dois episódios determinísticos garantem que a rotina de replay teste
    # ações/recompensas variadas mesmo se o agente aleatório escolher sempre 0.
    memory.store_experience(
        [1.0, 0.0, 0.0, 0.0], 0, 1.0, task_id="replay_smoke",
        learning_reward=0.75, truncated=True,
    )
    memory.store_experience(
        [0.0, 1.0, 0.0, 0.0], 1, 1.0, task_id="replay_smoke",
        learning_reward=-0.25,
    )
    experience_count = len(memory.episodic_buffer)
    assert experience_count > 0, "Nenhuma experiência foi armazenada."
    assert all("next_state" in item and "done" in item for item in memory.episodic_buffer)
    assert all("learning_reward" in item and "truncated" in item for item in memory.episodic_buffer)

    print("\n[FASE B] Replay offline e consolidação")
    engram_file = "test_engram.pt"
    expected_reward_baseline = agent.reward_baseline
    expected_baseline_alpha = agent.reward_baseline_alpha
    expected_use_td_critic = agent.use_td_critic
    weights_before_replay = [layer.sparse_weights.values().clone() for layer in engine.synapses]
    replay_stats = agent.sleep(
        memory_name=engram_file,
        replay_batch_size=min(8, experience_count),
        replay_reward_scale=0.5,
    )
    expected_critic_state = agent.critic.state_dict()
    assert replay_stats["replayed"] > 0, "O ciclo de sono não reproduziu episódios."
    assert len(memory.episodic_buffer) == experience_count, "O sono apagou o buffer episódico."
    replay_weight_change = sum(
        float((layer.sparse_weights.values() - before).abs().sum().item())
        for layer, before in zip(engine.synapses, weights_before_replay)
    )
    assert replay_weight_change > 0.0, "Replay não produziu atualização sináptica."

    expected_weights = [
        (layer.sparse_weights.indices().clone(), layer.sparse_weights.values().clone())
        for layer in engine.synapses
    ]

    print("[FASE C] Simulando desligamento e restaurando snapshot completo")
    for layer in engine.synapses:
        weights = layer.sparse_weights.coalesce()
        layer.sparse_weights = torch.sparse_coo_tensor(
            weights.indices(), torch.zeros_like(weights.values()),
            size=weights.size(), device=engine.device,
            check_invariants=True,
        ).coalesce()
    memory.clear_short_term_memory()
    simulator.time_steps = 2
    encoder.max_rate = 1.0
    agent.reward_baseline = 123.0
    agent.reward_baseline_alpha = 0.5
    agent.use_td_critic = False
    agent.critic.weights = np.full_like(agent.critic.weights, 99.0)

    assert agent.restore_memory(engram_file), "Falha ao restaurar snapshot."
    assert len(memory.episodic_buffer) == experience_count, "Experiências não foram restauradas."
    assert simulator.time_steps == 10, "Configuração temporal não foi restaurada."
    assert encoder.max_rate == 100.0, "Configuração do encoder não foi restaurada."
    assert agent.reward_baseline == expected_reward_baseline, "Baseline de recompensa não foi restaurado."
    assert agent.reward_baseline_alpha == expected_baseline_alpha, "Taxa do baseline não foi restaurada."
    assert agent.use_td_critic == expected_use_td_critic, "Modo do crítico não foi restaurado."
    assert agent.critic.state_dict() == expected_critic_state, "Crítico TD(0) não foi restaurado."
    for layer, (expected_indices, expected_values) in zip(engine.synapses, expected_weights):
        restored = layer.sparse_weights.coalesce()
        assert torch.equal(restored.indices(), expected_indices), "Topologia sináptica não foi restaurada."
        assert torch.equal(restored.values(), expected_values), "Pesos não foram restaurados exatamente."

    print(
        f"Replay: {replay_stats['replayed']} episódios; buffer preservado: "
        f"{len(memory.episodic_buffer)}; Δpesos L1={replay_weight_change:.6f}; "
        "pesos/topologia/runtime restaurados exatamente."
    )

    print("\n[FASE D] Verificando que o agente continua operando após restauração")
    telemetry.reset_logs()
    state = world.reset()
    for _ in range(10):
        action = agent.step(state)
        next_state, reward, done = world.step(action)
        telemetry.log_step(state, action, reward)
        state = next_state
        if done:
            break
    telemetry.print_summary("Episódio após restauração")
    print("\nTESTE INTEGRADO DE MEMÓRIA CONCLUÍDO COM SUCESSO")


if __name__ == "__main__":
    run_phase3_full_test()
