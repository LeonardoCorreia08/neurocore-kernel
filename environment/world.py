'''Modulo: world.py'''
import numpy as np

class VirtualWorld2D:
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.agent_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self.obstacles = [
            np.array([40.0, 40.0]),
            np.array([60.0, 70.0]),
            np.array([30.0, 80.0])
        ]

    def reset(self):
        self.agent_pos = np.random.rand(2) * self.width
        self.goal_pos = np.random.rand(2) * self.width
        return self.get_sensors()

    def get_sensors(self):
        dist_goal = (self.goal_pos - self.agent_pos) / self.width
        obs_detected = 0.0
        for obs in self.obstacles:
            if np.linalg.norm(self.agent_pos - obs) < 15.0:
                obs_detected = 1.0
        return np.array([dist_goal[0], dist_goal[1], obs_detected, 1.0], dtype=np.float32)

    def step(self, action_id: int):
        step_size = 5.0
        if action_id == 0: self.agent_pos[1] += step_size
        elif action_id == 1: self.agent_pos[1] -= step_size
        elif action_id == 2: self.agent_pos[0] -= step_size
        elif action_id == 3: self.agent_pos[0] += step_size

        self.agent_pos = np.clip(self.agent_pos, 0, self.width)
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        hit_obstacle = any(np.linalg.norm(self.agent_pos - obs) < 4.0 for obs in self.obstacles)

        reward, done = 0.0, False
        if distance_to_goal < 5.0:
            reward, done = 1.0, True
        elif hit_obstacle:
            reward, done = -0.5, True

        return self.get_sensors(), reward, done