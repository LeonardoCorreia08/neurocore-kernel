'''Modulo: core/simulation.py'''
import random

import torch

from core.spike_engine import SpikeEngine


class TemporalSimulator:
    def __init__(self, engine: SpikeEngine, time_steps: int = 20):
        """Executa ticks temporais e decodifica a atividade da camada motora."""
        if time_steps <= 0:
            raise ValueError("time_steps precisa ser positivo")
        self.engine = engine
        self.time_steps = int(time_steps)
        self.last_motor_spike_counts = torch.zeros(
            self.engine.neurons[-1].num_neurons, device=self.engine.device
        )
        self.last_action_silent = False

    def predict_action(
        self,
        sensory_array: list[float] | torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """Escolhe ação epsilon-greedy; empates/silêncio usam desempate aleatório.

        `epsilon=0` desliga exploração epsilon-greedy, mas empates continuam
        aleatórios para impedir que uma saída totalmente silenciosa favoreça
        sistematicamente o índice 0. Semeie o RNG Python para reprodução.
        """
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon precisa estar no intervalo [0, 1]")

        if isinstance(sensory_array, torch.Tensor):
            sensory_current = sensory_array.clone().detach().float()
        else:
            sensory_current = torch.tensor(sensory_array, dtype=torch.float32)
        sensory_current = sensory_current.to(self.engine.device)

        num_motor_neurons = self.engine.neurons[-1].num_neurons
        motor_spike_counts = torch.zeros(
            num_motor_neurons,
            device=self.engine.device,
            dtype=sensory_current.dtype,
        )
        for _ in range(self.time_steps):
            all_layer_spikes = self.engine(sensory_current)
            motor_spike_counts += all_layer_spikes[-1].view(-1)

        self.last_motor_spike_counts = motor_spike_counts.detach().clone()
        self.last_action_silent = bool(motor_spike_counts.sum().item() == 0.0)

        if random.random() < float(epsilon):
            return random.randrange(num_motor_neurons)

        max_count = motor_spike_counts.max()
        tied_actions = torch.nonzero(
            motor_spike_counts == max_count, as_tuple=False
        ).view(-1).detach().cpu().tolist()
        return int(random.choice(tied_actions))

    def process_feedback(self, reward: float, action: int | None = None):
        """Recebe o modulador de recompensa e o encaminha ao motor."""
        self.engine.apply_global_reward(float(reward), target_action=action)

    def reset_for_new_episode(self):
        """Zera membranas e rastros, preservando pesos e configuração aprendidos."""
        self.engine.reset_all_states()
        self.last_motor_spike_counts.zero_()
        self.last_action_silent = False
