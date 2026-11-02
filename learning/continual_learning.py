'''Modulo: learning/continual_learning.py'''
import numpy as np

from core.config_loader import get_config_value
from core.simulation import TemporalSimulator
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder


class ContinualLearningAgent:
    def __init__(self, simulator: TemporalSimulator, memory: MemoryManager, encoder: SensorEncoder, config: dict | None = None):
        """
        O Agente Cognitivo de Aprendizado Contínuo.
        Integra Percepção, Atenção, Memória e a SNN (Motor) para tomar decisões no Mundo.
        """
        self.simulator = simulator
        self.memory = memory
        self.encoder = encoder
        self.agent_danger_threshold = get_config_value(config, "brain.attention.agent_danger_threshold", 0.5)
        self.agent_target_attenuation = get_config_value(config, "brain.attention.agent_target_attenuation", 0.1)

    def apply_attention(self, sensory_state: np.ndarray) -> np.ndarray:
        """Filtro Cognitivo: Foca na sobrevivência se houver perigo iminente."""
        attended_state = sensory_state.copy()

        obs_danger = attended_state[2]
        wall_danger = attended_state[3]

        # Suprime a vontade de buscar o alvo se estiver prestes a bater
        if obs_danger > self.agent_danger_threshold or wall_danger > self.agent_danger_threshold:
            attended_state[0] *= self.agent_target_attenuation
            attended_state[1] *= self.agent_target_attenuation

        return attended_state

    def step(self, raw_sensory_state: np.ndarray) -> int:
        """Ciclo Cognitivo (O-O-D-A): Observar, Orientar, Decidir, Agir"""
        attended_state = self.apply_attention(raw_sensory_state)

        # Converte a realidade analógica em sinais elétricos (Corrente)
        neural_input = self.encoder.encode_current(attended_state)

        # O simulador pulsa o cérebro no tempo e vota numa ação (Rate Coding)
        action_id = self.simulator.predict_action(neural_input)

        return action_id

    def process_feedback(self, state: np.ndarray, action: int, reward: float):
        """Processa as consequências das ações (Dor/Prazer e Registo)."""
        self.memory.store_experience(state.tolist(), action, reward)
        self.simulator.process_feedback(reward)

    def sleep(self):
        """Ciclo de Sono: Consolida a memória e reinicia para o próximo dia/episódio."""
        self.memory.consolidate_memory()
        self.memory.clear_short_term_memory()
        self.simulator.reset_for_new_episode()
