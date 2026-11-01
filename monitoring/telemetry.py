'''Modulo: monitoring/telemetry.py'''
import numpy as np

class TelemetryManager:
    def __init__(self):
        """
        Item 20: Padronizar métricas, telemetria e registro da atividade.
        """
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
        
        # Conta quantas vezes cada ação (0 a 3) foi escolhida
        act_dist = np.bincount(self.actions_taken, minlength=4).tolist()
        
        print("\n" + "="*40)
        print(f"📊 TELEMETRIA: {episode_name}")
        print("="*40)
        print(f"Passos Sobrevividos : {steps}")
        print(f"Recompensa Total    : {total_reward:.2f}")
        print(f"Distribuição de Ações (0 a 3): {act_dist}")
        print("="*40 + "\n")
        
    def reset_logs(self):
        self.episode_rewards.clear()
        self.actions_taken.clear()
        self.sensory_history.clear()