"""Avaliação reproduzível do aprendizado por reforço no VirtualWorld2D.

Compara agente sem treino, após R-STDP online e após replay de sleep() com
baseline aleatório. É um benchmark pequeno de navegação, não uma prova de
capacidade geral.
"""

from __future__ import annotations

import json
import random
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from core.simulation import TemporalSimulator
from core.spike_engine import SpikeEngine
from environment.world import VirtualWorld2D
from learning.continual_learning import ContinualLearningAgent
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder

SEEDS = (210, 211, 212, 213, 214)
TRAIN_EPISODES = 40
MAX_STEPS = 50
TIME_STEPS = 10
EVAL_EPISODES = 20
USE_TD_CRITIC = False


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_learning(engine, enabled: bool) -> tuple[bool, bool, bool]:
    previous = (engine.enable_stdp, engine.enable_rstdp, engine.enable_homeostasis)
    engine.enable_stdp = enabled
    engine.enable_rstdp = enabled
    engine.enable_homeostasis = enabled
    return previous


def fixed_eval_cases(seed: int, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    # Use an independent, reproducible RNG so baseline, pre-training and
    # post-training evaluations share exactly the same maps/start-goal pairs.
    rng = np.random.RandomState(seed + 100_000)
    world = VirtualWorld2D(sensor_mode="directional")
    cases = []
    while len(cases) < count:
        start = rng.rand(2) * world.width
        goal = rng.rand(2) * world.width
        if np.linalg.norm(start - goal) <= 20.0:
            continue
        if any(np.linalg.norm(start - obs) < world.obstacle_radius for obs in world.obstacles):
            continue
        if any(
            np.linalg.norm(goal - obs) < world.obstacle_radius + world.goal_radius
            for obs in world.obstacles
        ):
            continue
        cases.append((start.astype(np.float64), goal.astype(np.float64)))
    return cases


def evaluate_agent(agent, cases, seed: int) -> dict:
    random.seed(seed + 300_000)
    world = VirtualWorld2D(sensor_mode="directional")
    counts = np.zeros(4, dtype=np.int64)
    rewards, steps, successes = [], [], []
    silent_decisions = 0
    original_flags = set_learning(agent.simulator.engine, False)
    try:
        for start, goal in cases:
            world.agent_pos = start.copy()
            world.goal_pos = goal.copy()
            agent.simulator.reset_for_new_episode()
            state = world.get_sensors()
            total_reward = 0.0
            successful = False
            step_count = 0
            for step_count in range(1, MAX_STEPS + 1):
                action = agent.step(state)
                silent_decisions += int(agent.simulator.last_action_silent)
                counts[action] += 1
                state, reward, done = world.step(action)
                total_reward += reward
                if done:
                    successful = reward > 0.0
                    break
            rewards.append(total_reward)
            steps.append(step_count)
            successes.append(float(successful))
    finally:
        (agent.simulator.engine.enable_stdp,
         agent.simulator.engine.enable_rstdp,
         agent.simulator.engine.enable_homeostasis) = original_flags
        agent.simulator.reset_for_new_episode()
    return {
        "success_rate": float(np.mean(successes)),
        "successes": int(sum(successes)),
        "episodes": len(cases),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "action_counts": counts.tolist(),
        "action_frequencies": (counts / max(int(counts.sum()), 1)).tolist(),
        "silent_decisions": silent_decisions,
    }


def evaluate_random(cases, seed: int) -> dict:
    world = VirtualWorld2D(sensor_mode="directional")
    rng = random.Random(seed + 200_000)
    rewards, steps, successes = [], [], []
    counts = np.zeros(4, dtype=np.int64)
    for start, goal in cases:
        world.agent_pos = start.copy()
        world.goal_pos = goal.copy()
        state = world.get_sensors()
        total_reward = 0.0
        successful = False
        step_count = 0
        for step_count in range(1, MAX_STEPS + 1):
            action = rng.randrange(4)
            counts[action] += 1
            state, reward, done = world.step(action)
            total_reward += reward
            if done:
                successful = reward > 0.0
                break
        rewards.append(total_reward)
        steps.append(step_count)
        successes.append(float(successful))
    return {
        "success_rate": float(np.mean(successes)),
        "successes": int(sum(successes)),
        "episodes": len(cases),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "action_counts": counts.tolist(),
        "action_frequencies": (counts / max(int(counts.sum()), 1)).tolist(),
    }


def train_one_seed(seed: int) -> dict:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = VirtualWorld2D(sensor_mode="directional")
    engine = SpikeEngine(
        layer_sizes=[6, 32, 4],
        density=0.25,
        device=str(device),
        enable_stdp=True,
        enable_rstdp=True,
        enable_homeostasis=True,
    )
    simulator = TemporalSimulator(engine, time_steps=TIME_STEPS)
    with tempfile.TemporaryDirectory(prefix="neurocore-rl-eval-") as save_dir:
        memory = MemoryManager(engine, save_dir=save_dir, max_buffer_size=1000)
        agent = ContinualLearningAgent(
            simulator,
            memory,
            SensorEncoder(max_rate=100.0),
            use_td_critic=USE_TD_CRITIC,
        )
        cases = fixed_eval_cases(seed, EVAL_EPISODES)

        random_baseline = evaluate_random(cases, seed)
        initial = evaluate_agent(agent, cases, seed)

        episode_rewards = []
        training_action_counts = np.zeros(4, dtype=np.int64)
        training_silent_decisions = 0
        epsilon_start, epsilon_end = 1.0, 0.05
        for episode in range(TRAIN_EPISODES):
            simulator.reset_for_new_episode()
            state = world.reset()
            progress = episode / max(TRAIN_EPISODES - 1, 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
            total_reward = 0.0
            for step in range(1, MAX_STEPS + 1):
                action = agent.step(state, epsilon=epsilon)
                training_silent_decisions += int(simulator.last_action_silent)
                training_action_counts[action] += 1
                next_state, reward, done = world.step(action)
                agent.process_feedback(
                    state, action, reward,
                    next_state=next_state,
                    done=done,
                    task_id="navigation",
                    truncated=(step == MAX_STEPS and not done),
                )
                total_reward += reward
                state = next_state
                if done:
                    break
            episode_rewards.append(total_reward)

        online = evaluate_agent(agent, cases, seed)
        replay_stats = agent.sleep(
            memory_name="assessment.pt",
            replay_batch_size=min(32, len(memory.episodic_buffer)),
            replay_reward_scale=0.5,
            task_id="navigation",
        )
        after_replay = evaluate_agent(agent, cases, seed)

        return {
            "seed": seed,
            "random_baseline": random_baseline,
            "untrained_agent": initial,
            "after_online_rstdp": online,
            "after_sleep_replay": after_replay,
            "training": {
                "episodes": TRAIN_EPISODES,
                "first_10_mean_reward": float(np.mean(episode_rewards[:10])),
                "last_10_mean_reward": float(np.mean(episode_rewards[-10:])),
                "all_mean_reward": float(np.mean(episode_rewards)),
                "first_10_successes": int(sum(r > 0 for r in episode_rewards[:10])),
                "last_10_successes": int(sum(r > 0 for r in episode_rewards[-10:])),
                "action_counts": training_action_counts.tolist(),
                "silent_decisions": training_silent_decisions,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "stored_experiences": len(memory.episodic_buffer),
                "replay_count": replay_stats["replayed"],
                "td_critic_enabled": agent.use_td_critic,
                "mean_replay_learning_signal": replay_stats.get(
                    "mean_learning_signal", replay_stats["mean_reward"]
                ),
            },
        }


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 2.776 * sd / (len(values) ** 0.5) if len(values) == 5 else 1.96 * sd / (len(values) ** 0.5)
    return mean, mean - margin, mean + margin


def summarize(records: list[dict]) -> dict:
    result = {}
    for phase in ("random_baseline", "untrained_agent", "after_online_rstdp", "after_sleep_replay"):
        result[phase] = {}
        for metric in ("success_rate", "mean_reward", "mean_steps"):
            vals = [float(r[phase][metric]) for r in records]
            mean, low, high = mean_ci95(vals)
            if metric == "success_rate":
                mean, low, high = 100 * mean, 100 * low, 100 * high
                low, high = max(0.0, low), min(100.0, high)
            result[phase][metric] = {"mean": mean, "ci95_low": low, "ci95_high": high}
    return result


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("NeuroCore RL assessment | epsilon-greedy, reset, sensores direcionais e baseline móvel")
    print(
        f"device={device}; seeds={len(SEEDS)}; treino={TRAIN_EPISODES} episódios/seed; "
        f"avaliação={EVAL_EPISODES} mapas/seed; limite={MAX_STEPS} passos"
    )
    print("Success = reward terminal > 0; avaliação usa os mesmos mapas nas condições comparadas.")
    records = []
    for seed in SEEDS:
        row = train_one_seed(seed)
        records.append(row)
        print(
            f"seed={seed}: sucesso random={row['random_baseline']['successes']}/{EVAL_EPISODES}, "
            f"inicial={row['untrained_agent']['successes']}/{EVAL_EPISODES}, "
            f"R-STDP={row['after_online_rstdp']['successes']}/{EVAL_EPISODES}, "
            f"após replay={row['after_sleep_replay']['successes']}/{EVAL_EPISODES}; "
            f"recompensa treino 1ª/última década="
            f"{row['training']['first_10_mean_reward']:.2f}/"
            f"{row['training']['last_10_mean_reward']:.2f}; "
            f"ações no treino={row['training']['action_counts']}"
        )

    summary = summarize(records)
    print("\nMÉDIAS POR SEMENTE (IC t aproximado de 95%, n=5)")
    for phase, metrics in summary.items():
        print(phase)
        print(
            "  success %: "
            f"{metrics['success_rate']['mean']:.1f} "
            f"[{metrics['success_rate']['ci95_low']:.1f}, {metrics['success_rate']['ci95_high']:.1f}]"
        )
        for metric in ("mean_reward", "mean_steps"):
            val = metrics[metric]
            print(f"  {metric}: {val['mean']:.3f} [{val['ci95_low']:.3f}, {val['ci95_high']:.3f}]")

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seed_list": list(SEEDS),
            "train_episodes_per_seed": TRAIN_EPISODES,
            "eval_episodes_per_seed": EVAL_EPISODES,
            "max_steps": MAX_STEPS,
            "time_steps_per_decision": TIME_STEPS,
            "sensor_mode": "directional",
            "input_features": 6,
            "td_critic_enabled": USE_TD_CRITIC,
            "success_definition": "environment terminal reward > 0",
            "test_maps_shared_across_random_untrained_trained_post_replay_conditions": True,
            "notes": [
                "Training resets dynamic neural state between episodes while preserving learned synaptic weights.",
                "Training uses linear epsilon-greedy exploration from 1.0 to 0.05; evaluation uses epsilon=0 with randomized ties.",
                "Default R-STDP modulation uses a moving reward baseline; optional TD critic can be enabled with USE_TD_CRITIC.",
                "Six signed directional features encode goal, nearest-obstacle repulsion and wall repulsion; goal samples avoid obstacle clearance zones.",
                "Small five-seed benchmark; confidence intervals are approximate and environment is synthetic.",
            ],
        },
        "summary": summary,
        "runs": records,
    }
    result_name = (
        "reinforcement_learning_assessment_td_critic.json"
        if USE_TD_CRITIC
        else "reinforcement_learning_assessment_directional_baseline.json"
    )
    path = Path(__file__).resolve().parent / "results" / result_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResultados detalhados salvos em: {path}")


if __name__ == "__main__":
    main()
