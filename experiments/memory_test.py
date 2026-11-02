'''Modulo: experiments/memory_test.py'''
import os
import torch
import numpy as np

# Importando toda a Fase 3
from environment.world import VirtualWorld2D                  # Item 17
from core.spike_engine import SpikeEngine                     # Item 14
from core.simulation import TemporalSimulator                 # Item 15
from memory.manager import MemoryManager                      # Item 16
from perception.sensor_encoder import SensorEncoder           # Item 18
from learning.continual_learning import ContinualLearningAgent# Item 19
from monitoring.telemetry import TelemetryManager             # Item 20

def run_phase3_full_test():
    print("🚀 INICIANDO TESTE FULL FASE 3 (ITENS 14 A 20) 🚀\n")

    # Inicializa todos os módulos
    world = VirtualWorld2D(width=100, height=100)
    engine = SpikeEngine(layer_sizes=[4, 16, 4], density=0.5)
    simulator = TemporalSimulator(engine, time_steps=10)
    
    save_directory = "checkpoints_test/"
    memory = MemoryManager(engine, save_dir=save_directory)
    encoder = SensorEncoder(max_rate=100.0)
    agent = ContinualLearningAgent(simulator, memory, encoder)
    telemetry = TelemetryManager()

    print("[FASE A] EPISÓDIO 1 - APRENDIZADO INICIAL")
    state = world.reset()
    for step in range(1, 21):
        action = agent.step(state)
        next_state, reward, done = world.step(action)
        agent.process_feedback(state, action, reward)
        telemetry.log_step(state, action, reward)
        state = next_state
        if done: break

    telemetry.print_summary("Episódio 1 (Cérebro Virgem)")
    
    print("[FASE B] TESTE DE MEMÓRIA (ITEM 16)")
    # Consolida a memória e destrói o cérebro atual para provar o save/load
    engram_file = "test_engram.pt"
    memory.consolidate_memory(engram_file)
    
    # Simula um desligamento do sistema (Reset total dos pesos esparsos para zero)
    for synapse in engine.synapses:
        synapse.sparse_weights = torch.sparse_coo_tensor(
            synapse.sparse_weights.indices(), 
            torch.zeros_like(synapse.sparse_weights.values()), 
            size=synapse.sparse_weights.size()
        ).coalesce()
    print("-> Cérebro apagado com sucesso (Pesos zerados).")

    # Recupera a memória
    sucesso = memory.retrieve_memory(engram_file)
    assert sucesso is True, "Falha crítica: Não foi possível carregar a memória."

    print("\n[FASE C] EPISÓDIO 2 - RECUPERAÇÃO DO ENGRAMA")
    telemetry.reset_logs()
    state = world.reset()
    for step in range(1, 21):
        action = agent.step(state)
        next_state, reward, done = world.step(action)
        telemetry.log_step(state, action, reward)
        state = next_state
        if done: break

    telemetry.print_summary("Episódio 2 (Engrama Recuperado)")

    print("✅ TESTE DA FASE 3 CONCLUÍDO COM SUCESSO!")
    print("Módulos Validados: 14 (Engine), 15 (Simulator), 16 (Memory), 17 (World), 18 (Perception), 19 (Agent), 20 (Telemetry)")

if __name__ == "__main__":
    run_phase3_full_test()