"""Agente contínuo com memória episódica e replay de experiências."""

import inspect

import numpy as np

from core.simulation import TemporalSimulator
from learning.td_critic import LinearTDCritic
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder


def _supports_keyword(callable_object, keyword: str) -> bool:
    """Detecta extensões opcionais ao lidar com checkouts de versões mistas."""
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


class ContinualLearningAgent:
    def __init__(
        self,
        simulator: TemporalSimulator,
        memory: MemoryManager,
        encoder: SensorEncoder,
        reward_baseline_alpha: float = 0.01,
        critic_gamma: float = 0.95,
        critic_learning_rate: float = 0.05,
        use_td_critic: bool = False,
    ):
        if not 0.0 < reward_baseline_alpha <= 1.0:
            raise ValueError("reward_baseline_alpha precisa estar em (0, 1]")
        self.simulator = simulator
        self.memory = memory
        self.encoder = encoder
        self.reward_baseline_alpha = float(reward_baseline_alpha)
        self.reward_baseline = 0.0
        self.use_td_critic = bool(use_td_critic)
        self.critic = LinearTDCritic(
            gamma=critic_gamma,
            learning_rate=critic_learning_rate,
        )

    def apply_attention(self, sensory_state: np.ndarray) -> np.ndarray:
        """Keep the state representation intact; directional features carry signed cues."""
        return np.asarray(sensory_state, dtype=np.float32).copy()

    def step(self, raw_sensory_state: np.ndarray, epsilon: float = 0.0) -> int:
        attended_state = self.apply_attention(raw_sensory_state)
        neural_input = self.encoder.encode_current(attended_state)
        if _supports_keyword(self.simulator.predict_action, "epsilon"):
            return self.simulator.predict_action(neural_input, epsilon=epsilon)
        return self.simulator.predict_action(neural_input)

    def process_feedback(
        self,
        state,
        action: int,
        reward: float,
        next_state=None,
        done: bool = False,
        task_id: str | int | None = None,
        truncated: bool = False,
    ) -> None:
        """Store the transition and modulate R-STDP with an explicit signal.

        The default uses a moving reward baseline. A linear TD(0) critic is
        opt-in because its short benchmark did not improve success. When TD is
        enabled, its error is used online and recomputed from stored transitions
        during replay. Legacy calls without a next state use the baseline.
        """
        raw_reward = float(reward)
        if self.use_td_critic and (next_state is not None or done):
            learning_signal = self.critic.update(
                state,
                raw_reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
            )
        else:
            learning_signal = raw_reward - self.reward_baseline
            self.reward_baseline += self.reward_baseline_alpha * learning_signal

        store_kwargs = {
            "next_state": next_state,
            "done": done,
            "task_id": task_id,
        }
        if _supports_keyword(self.memory.store_experience, "learning_reward"):
            store_kwargs["learning_reward"] = learning_signal
        if _supports_keyword(self.memory.store_experience, "truncated"):
            store_kwargs["truncated"] = truncated
        self.memory.store_experience(state, action, raw_reward, **store_kwargs)

        if _supports_keyword(self.simulator.process_feedback, "action"):
            self.simulator.process_feedback(learning_signal, action=action)
        else:
            self.simulator.process_feedback(learning_signal)

    def replay_experiences(
        self,
        batch_size: int = 32,
        reward_scale: float = 0.5,
        task_id: str | int | None = None,
    ) -> dict:
        """Reactivate stored transitions using teacher-forced motor spikes."""
        experiences = self.memory.sample_experiences(
            batch_size=batch_size,
            task_id=task_id,
            balanced=True,
        )
        if not experiences:
            return {"replayed": 0, "mean_reward": 0.0}

        engine = self.simulator.engine
        replayed_signals = []
        for experience in experiences:
            raw_state = np.asarray(experience["state"], dtype=np.float32)
            attended_state = self.apply_attention(raw_state)
            neural_input = self.encoder.encode_current(attended_state).to(engine.device)
            action = int(experience["action"])

            engine.reset_all_states()
            for _ in range(self.simulator.time_steps):
                if _supports_keyword(engine.forward, "teacher_action"):
                    engine(neural_input, teacher_action=action)
                else:
                    engine(neural_input)

            raw_reward = float(experience["reward"])
            next_state = experience.get("next_state")
            done = bool(experience.get("done", False))
            if self.use_td_critic and (next_state is not None or done):
                learning_signal = self.critic.update(
                    raw_state,
                    raw_reward,
                    next_state=next_state,
                    done=done,
                    truncated=bool(experience.get("truncated", False)),
                )
            else:
                learning_signal = float(
                    experience.get("learning_reward", raw_reward)
                )
            replay_reward = learning_signal * float(reward_scale)
            if _supports_keyword(engine.apply_global_reward, "target_action"):
                engine.apply_global_reward(replay_reward, target_action=action)
            else:
                engine.apply_global_reward(replay_reward)
            replayed_signals.append(replay_reward)

        engine.reset_all_states()
        return {
            "replayed": len(experiences),
            "mean_reward": float(np.mean(replayed_signals)),
            "mean_learning_signal": float(np.mean(replayed_signals)),
            "actions": [int(item["action"]) for item in experiences],
        }

    def _runtime_metadata(self) -> dict:
        return {
            "time_steps": int(self.simulator.time_steps),
            "encoder_max_rate": float(self.encoder.max_rate),
            "encoder_dt": float(self.encoder.dt),
            "reward_baseline": float(self.reward_baseline),
            "reward_baseline_alpha": float(self.reward_baseline_alpha),
            "use_td_critic": bool(self.use_td_critic),
            "td_critic": self.critic.state_dict(),
        }

    def restore_memory(self, memory_name: str = "latest_brain_engram.pt") -> bool:
        restored = self.memory.retrieve_memory(memory_name)
        if not restored:
            return False
        metadata = self.memory.restored_runtime
        if "time_steps" in metadata:
            self.simulator.time_steps = int(metadata["time_steps"])
        if "encoder_max_rate" in metadata:
            self.encoder.max_rate = float(metadata["encoder_max_rate"])
        if "encoder_dt" in metadata:
            self.encoder.dt = float(metadata["encoder_dt"])
        if "reward_baseline" in metadata:
            self.reward_baseline = float(metadata["reward_baseline"])
        if "reward_baseline_alpha" in metadata:
            self.reward_baseline_alpha = float(metadata["reward_baseline_alpha"])
        if "use_td_critic" in metadata:
            self.use_td_critic = bool(metadata["use_td_critic"])
        if "td_critic" in metadata:
            self.critic.load_state_dict(metadata["td_critic"])
        return True

    def save_checkpoint(
        self,
        memory_name: str = "latest_brain_engram.pt",
        runtime_metadata: dict | None = None,
    ) -> str:
        """Save a clean episode boundary without applying another replay."""
        self.simulator.reset_for_new_episode()
        metadata = self._runtime_metadata()
        if runtime_metadata:
            metadata.update(runtime_metadata)
        return self.memory.consolidate_memory(
            memory_name,
            runtime_metadata=metadata,
        )

    def sleep(
        self,
        memory_name: str = "latest_brain_engram.pt",
        replay_batch_size: int = 32,
        replay_reward_scale: float = 0.5,
        task_id: str | int | None = None,
        runtime_metadata: dict | None = None,
    ) -> dict:
        """Replay selected episodes and checkpoint the consolidated boundary."""
        replay_stats = self.replay_experiences(
            batch_size=replay_batch_size,
            reward_scale=replay_reward_scale,
            task_id=task_id,
        )
        self.save_checkpoint(
            memory_name,
            runtime_metadata=runtime_metadata,
        )
        return replay_stats
