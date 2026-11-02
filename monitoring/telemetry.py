'''Modulo: monitoring/telemetry.py'''
import numpy as np


def _settings(config):
    if not config:
        return {}
    return config.get("monitoring", config).get("telemetry", {})


class TelemetryManager:
    def __init__(self, action_count: int = 4, min_action_id: int = 0, max_action_id: int = 3, config: dict = None):
        """Mantém dados de monitoramento de passos, separados da homeostase."""
        settings = _settings(config)
        self.action_count = settings.get("action_count", action_count)
        self.min_action_id = settings.get("min_action_id", min_action_id)
        self.max_action_id = settings.get("max_action_id", max_action_id)
        self.episode_rewards = []
        self.actions_taken = []
        self.sensory_history = []

    def log_step(self, state: np.ndarray, action: int, reward: float):
        """Regista os dados de cada passo de tempo (Tick)."""
        self.sensory_history.append(state)
        self.actions_taken.append(action)
        self.episode_rewards.append(reward)

    def print_summary(self, episode_name: str):
        """Gera um relatório estatístico do comportamento do agente."""
        total_reward = sum(self.episode_rewards)
        steps = len(self.actions_taken)
        valid_actions = [a - self.min_action_id for a in self.actions_taken if self.min_action_id <= a <= self.max_action_id]
        act_dist = np.bincount(valid_actions, minlength=self.action_count).tolist()
        print("\n" + "=" * 40)
        print(f"📊 TELEMETRIA: {episode_name}")
        print("=" * 40)
        print(f"Passos Sobrevividos : {steps}")
        print(f"Recompensa Total    : {total_reward:.2f}")
        print(f"Distribuição de Ações ({self.min_action_id} a {self.max_action_id}): {act_dist}")
        print("=" * 40 + "\n")

    def reset_logs(self):
        self.episode_rewards.clear()
        self.actions_taken.clear()
        self.sensory_history.clear()
