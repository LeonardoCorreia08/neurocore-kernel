'''Modulo: environment/world.py'''
import numpy as np


class VirtualWorld2D:
    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        sensor_mode: str = "compact",
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width e height precisam ser positivos")
        if sensor_mode not in {"compact", "directional"}:
            raise ValueError("sensor_mode deve ser 'compact' ou 'directional'")
        self.width = float(width)
        self.height = float(height)
        self.sensor_mode = sensor_mode
        self.agent_pos = np.zeros(2, dtype=np.float64)
        self.goal_pos = np.zeros(2, dtype=np.float64)

        self.obstacles = [
            np.array([40.0, 40.0]),
            np.array([60.0, 70.0]),
            np.array([30.0, 80.0]),
        ]
        self.step_size = 5.0
        self.goal_radius = 5.0
        self.obstacle_radius = 4.0
        self.sensor_range = 20.0

    def _is_obstacle_safe(self, position: np.ndarray, clearance: float = 0.0) -> bool:
        min_distance = self.obstacle_radius + clearance
        return all(np.linalg.norm(position - obs) >= min_distance for obs in self.obstacles)

    def reset(self) -> np.ndarray:
        """Sample agent and goal away from obstacles and away from each other."""
        bounds = np.array([self.width, self.height], dtype=np.float64)
        while True:
            agent_pos = np.random.rand(2) * bounds
            goal_pos = np.random.rand(2) * bounds
            far_enough = np.linalg.norm(agent_pos - goal_pos) > 20.0
            agent_safe = self._is_obstacle_safe(agent_pos)
            goal_safe = self._is_obstacle_safe(goal_pos, clearance=self.goal_radius)
            if far_enough and agent_safe and goal_safe:
                self.agent_pos = agent_pos
                self.goal_pos = goal_pos
                return self.get_sensors()

    def get_sensors(self) -> np.ndarray:
        """Return compact legacy sensors or a six-value directional state.

        Directional order: goal dx/dy, obstacle repulsion dx/dy, wall
        repulsion x/y. Values are signed and normalized to approximately
        [-1, 1], so the policy can distinguish which way to move.
        """
        goal_delta = self.goal_pos - self.agent_pos
        goal_signal = goal_delta / np.array([self.width, self.height])

        obstacle_distances = [
            np.linalg.norm(self.agent_pos - obstacle) for obstacle in self.obstacles
        ]
        nearest_index = int(np.argmin(obstacle_distances))
        nearest_delta = self.obstacles[nearest_index] - self.agent_pos
        nearest_distance = float(obstacle_distances[nearest_index])
        obstacle_danger = float(np.clip(1.0 - nearest_distance / self.sensor_range, 0.0, 1.0))
        if nearest_distance > 1e-8:
            obstacle_repulsion = -(nearest_delta / nearest_distance) * obstacle_danger
        else:
            obstacle_repulsion = np.zeros(2, dtype=np.float64)

        left_dist = self.agent_pos[0]
        right_dist = self.width - self.agent_pos[0]
        bottom_dist = self.agent_pos[1]
        top_dist = self.height - self.agent_pos[1]
        if left_dist <= right_dist:
            wall_x = max(0.0, 1.0 - left_dist / self.sensor_range)
        else:
            wall_x = -max(0.0, 1.0 - right_dist / self.sensor_range)
        if bottom_dist <= top_dist:
            wall_y = max(0.0, 1.0 - bottom_dist / self.sensor_range)
        else:
            wall_y = -max(0.0, 1.0 - top_dist / self.sensor_range)

        if self.sensor_mode == "directional":
            return np.asarray(
                [*goal_signal, *obstacle_repulsion, wall_x, wall_y], dtype=np.float32
            )

        wall_danger = max(abs(wall_x), abs(wall_y))
        return np.asarray(
            [goal_signal[0], goal_signal[1], obstacle_danger, wall_danger],
            dtype=np.float32,
        )

    def step(self, action_id: int):
        """Apply one cardinal movement; return next sensors, reward and done."""
        action_id = int(action_id)
        if not 0 <= action_id < 4:
            raise ValueError("action_id precisa estar entre 0 e 3")

        if action_id == 0:
            self.agent_pos[1] += self.step_size  # Cima
        elif action_id == 1:
            self.agent_pos[1] -= self.step_size  # Baixo
        elif action_id == 2:
            self.agent_pos[0] -= self.step_size  # Esquerda
        else:
            self.agent_pos[0] += self.step_size  # Direita

        self.agent_pos = np.clip(
            self.agent_pos,
            np.array([0.0, 0.0]),
            np.array([self.width, self.height]),
        )

        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        hit_obstacle = any(
            np.linalg.norm(self.agent_pos - obstacle) < self.obstacle_radius
            for obstacle in self.obstacles
        )
        reward = -0.01
        done = False
        if distance_to_goal < self.goal_radius:
            reward = 1.0
            done = True
        elif hit_obstacle:
            reward = -0.5
            done = True
        return self.get_sensors(), reward, done
