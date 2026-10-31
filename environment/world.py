'''Módulo: world.py'''
import numpy as np

class VirtualWorld2D:
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.agent_pos = np.array([50.0, 50.0])
        self.target_pos = np.array([80.0, 80.0])
        self.obstacles = [
            np.array([30.0, 30.0]),
            np.array([70.0, 40.0]),
            np.array([40.0, 70.0])
        ]
        
    def reset(self):
        """Reinicializa o agente em posição aleatória na arena."""
        self.agent_pos = np.random.rand(2) * 80.0 + 10.0
        self._spawn_new_target()
        return self.get_sensors()

    def _spawn_new_target(self):
        self.target_pos = np.random.rand(2) * 80.0 + 10.0

    def get_sensors(self):
        """Retorna distâncias normalizadas para paredes, alvo e obstáculos."""
        # Distância normalizada para o alvo (dx, dy)
        target_vec = (self.target_pos - self.agent_pos) / 100.0
        
        # Proximidade das bordas (paredes)
        wall_dist_x = min(self.agent_pos[0], self.width - self.agent_pos[0]) / self.width
        wall_dist_y = min(self.agent_pos[1], self.height - self.agent_pos[1]) / self.height
        
        # Vetor de sensores de entrada (ex: 4 entradas sensoriais)
        sensors = np.array([target_vec[0], target_vec[1], wall_dist_x, wall_dist_y], dtype=np.float32)
        return sensors

    def step(self, action_id: int):
        """
        Executa a ação escolhida pelo agente:
        0: Cima, 1: Baixo, 2: Esquerda, 3: Direita
        """
        step_size = 3.0
        if action_id == 0:
            self.agent_pos[1] += step_size
        elif action_id == 1:
            self.agent_pos[1] -= step_size
        elif action_id == 2:
            self.agent_pos[0] -= step_size
        elif action_id == 3:
            self.agent_pos[0] += step_size

        # Limitar dentro da arena
        self.agent_pos = np.clip(self.agent_pos, 0.0, float(self.width))

        # Verificar colisão com o alvo (recompensa)
        distance_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = 0.0
        done = False

        if distance_to_target < 5.0:
            reward = 1.0
            done = True
            self._spawn_new_target()

        return self.get_sensors(), reward, done