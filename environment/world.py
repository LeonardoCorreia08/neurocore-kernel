'''Modulo: environment/world.py'''
import numpy as np


class VirtualWorld2D:
    def __init__(self, width: int = 100, height: int = 100, config=None):
        # Configuração é opcional: sem ela, todos os valores originais persistem.
        environment = config.get("environment", config) if isinstance(config, dict) else {}
        self.width = environment.get("width", width)
        self.height = environment.get("height", height)
        self.agent_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)

        configured_obstacles = environment.get("obstacles")
        if configured_obstacles is None:
            configured_obstacles = [[40.0, 40.0], [60.0, 70.0], [30.0, 80.0]]
        self.obstacles = [np.asarray(obstacle, dtype=float) for obstacle in configured_obstacles]

        self.step_size = environment.get("step_size", 5.0)
        self.goal_radius = environment.get("goal_radius", 5.0)
        self.obstacle_radius = environment.get("obstacle_radius", 4.0)
        self.sensor_range = environment.get("sensor_range", 20.0)
        self.spawn_min_goal_distance = environment.get("spawn_min_goal_distance", 20.0)
        self.max_spawn_attempts = environment.get("max_spawn_attempts", 10000)
        reward_config = environment.get("reward", {})
        self.reward_per_step = reward_config.get("per_step", -0.01)
        self.reward_goal = reward_config.get("goal", 1.0)
        self.reward_obstacle_collision = reward_config.get("obstacle_collision", -0.5)

    def reset(self):
        """Reinicia com spawn seguro ou falha claramente se não for possível."""
        for _ in range(self.max_spawn_attempts):
            self.agent_pos = np.array([
                np.random.rand() * self.width,
                np.random.rand() * self.height,
            ])
            self.goal_pos = np.array([
                np.random.rand() * self.width,
                np.random.rand() * self.height,
            ])

            dist_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
            hit_obs = any(
                np.linalg.norm(self.agent_pos - obs) < self.obstacle_radius
                for obs in self.obstacles
            )
            if dist_goal > self.spawn_min_goal_distance and not hit_obs:
                return self.get_sensors()

        raise RuntimeError(
            f"Não foi possível encontrar spawn válido em {self.max_spawn_attempts} tentativas; "
            "verifique dimensões, obstáculos e spawn_min_goal_distance."
        )

    def get_sensors(self) -> np.ndarray:
        """Gera quatro sinais contínuos para a rede neural."""
        dist_goal = (self.goal_pos - self.agent_pos) / np.array([self.width, self.height])

        min_obs_dist = min(
            (np.linalg.norm(self.agent_pos - obs) for obs in self.obstacles),
            default=float("inf"),
        )
        if min_obs_dist < self.sensor_range:
            obs_danger = 1.0 - (min_obs_dist / self.sensor_range)
        else:
            obs_danger = 0.0

        dist_x_wall = min(self.agent_pos[0], self.width - self.agent_pos[0])
        dist_y_wall = min(self.agent_pos[1], self.height - self.agent_pos[1])
        min_wall_dist = min(dist_x_wall, dist_y_wall)
        if min_wall_dist < self.sensor_range:
            wall_danger = 1.0 - (min_wall_dist / self.sensor_range)
        else:
            wall_danger = 0.0

        return np.array([dist_goal[0], dist_goal[1], obs_danger, wall_danger], dtype=np.float32)

    def step(self, action_id: int):
        # Transição de movimento
        if action_id == 0: self.agent_pos[1] += self.step_size   # Cima
        elif action_id == 1: self.agent_pos[1] -= self.step_size # Baixo
        elif action_id == 2: self.agent_pos[0] -= self.step_size # Esquerda
        elif action_id == 3: self.agent_pos[0] += self.step_size # Direita

        # Limita cada eixo à dimensão correspondente.
        self.agent_pos = np.clip(self.agent_pos, [0, 0], [self.width, self.height])

        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        hit_obstacle = any(
            np.linalg.norm(self.agent_pos - obs) < self.obstacle_radius
            for obs in self.obstacles
        )

        reward = self.reward_per_step
        done = False
        if distance_to_goal < self.goal_radius:
            reward = self.reward_goal
            done = True
        elif hit_obstacle:
            reward = self.reward_obstacle_collision
            done = True

        return self.get_sensors(), reward, done
