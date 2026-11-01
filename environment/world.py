'''Modulo: environment/world.py'''
import numpy as np

class VirtualWorld2D:
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.agent_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        
        # Obstáculos fixos no mapa
        self.obstacles = [
            np.array([40.0, 40.0]),
            np.array([60.0, 70.0]),
            np.array([30.0, 80.0])
        ]
        
        self.step_size = 5.0
        self.goal_radius = 5.0
        self.obstacle_radius = 4.0
        self.sensor_range = 20.0 # Alcance da "visão" do agente

    def reset(self):
        """
        Reinicia o ambiente com 'Safe Spawn' (Garante que o agente não nasce 
        dentro de um obstáculo ou já em cima do objetivo).
        """
        valid_spawn = False
        while not valid_spawn:
            self.agent_pos = np.random.rand(2) * self.width
            self.goal_pos = np.random.rand(2) * self.width
            
            # Verifica se nasceu longe do objetivo e fora dos obstáculos
            dist_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
            hit_obs = any(np.linalg.norm(self.agent_pos - obs) < self.obstacle_radius for obs in self.obstacles)
            
            if dist_goal > 20.0 and not hit_obs:
                valid_spawn = True
                
        return self.get_sensors()

    def get_sensors(self) -> np.ndarray:
        """
        Gera 4 sinais contínuos (Estado) para a Rede Neural.
        Sinais contínuos são fundamentais para SNNs, pois viram taxas de disparo (frequência).
        """
        # Sensor 1 e 2: Direção do Alvo (Normalizada entre -1.0 e 1.0)
        dist_goal = (self.goal_pos - self.agent_pos) / self.width
        
        # Sensor 3: Proximidade de Obstáculos (0.0 = Seguro, 1.0 = Colisão Iminente)
        min_obs_dist = min([np.linalg.norm(self.agent_pos - obs) for obs in self.obstacles])
        if min_obs_dist < self.sensor_range:
            # Transforma a distância num sinal que aumenta quanto mais perto chega
            obs_danger = 1.0 - (min_obs_dist / self.sensor_range)
        else:
            obs_danger = 0.0
            
        # Sensor 4: Proximidade das Paredes (Limites do Mundo)
        dist_x_wall = min(self.agent_pos[0], self.width - self.agent_pos[0])
        dist_y_wall = min(self.agent_pos[1], self.height - self.agent_pos[1])
        min_wall_dist = min(dist_x_wall, dist_y_wall)
        
        if min_wall_dist < self.sensor_range:
            wall_danger = 1.0 - (min_wall_dist / self.sensor_range)
        else:
            wall_danger = 0.0

        # Retorna o array de 4 dimensões (Alvo X, Alvo Y, Perigo Obstáculo, Perigo Parede)
        return np.array([dist_goal[0], dist_goal[1], obs_danger, wall_danger], dtype=np.float32)

    def step(self, action_id: int):
        # Transição de movimento
        if action_id == 0: self.agent_pos[1] += self.step_size   # Cima
        elif action_id == 1: self.agent_pos[1] -= self.step_size # Baixo
        elif action_id == 2: self.agent_pos[0] -= self.step_size # Esquerda
        elif action_id == 3: self.agent_pos[0] += self.step_size # Direita

        # Limita o movimento aos limites do mapa
        self.agent_pos = np.clip(self.agent_pos, 0, self.width)
        
        # Cálculos de física
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        hit_obstacle = any(np.linalg.norm(self.agent_pos - obs) < self.obstacle_radius for obs in self.obstacles)

        # Sistema de Recompensas
        reward = -0.01  # "Time penalty": Punição leve a cada passo para forçar o agente a ser rápido
        done = False
        
        if distance_to_goal < self.goal_radius:
            reward = 1.0  # Sucesso absoluto
            done = True
        elif hit_obstacle:
            reward = -0.5 # Punição dolorosa (Bateu)
            done = True

        return self.get_sensors(), reward, done