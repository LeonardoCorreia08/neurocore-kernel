"""teste v5 commit"""
from __future__ import annotations

import argparse
import random
texto novissimo
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.config_loader import get_config_value, load_brain_config
from core.spike_engine import SpikeEngine
from core.simulation import TemporalSimulator
from environment.world import VirtualWorld2D
from learning.continual_learning import ContinualLearningAgent
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder


def render_world(world, episode, step, ax, runtime):
    """Renderiza um quadro usando as opções centralizadas de runtime."""
    ax.clear()
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_title(f"NeuroCore | Episódio: {episode} | Passo: {step}")
    for obs in world.obstacles:
        ax.add_patch(plt.Circle(
            (obs[0], obs[1]), world.obstacle_radius, color="red", alpha=0.6
        ))
    ax.add_patch(plt.Circle(
        (world.goal_pos[0], world.goal_pos[1]), world.goal_radius,
        color="green", alpha=0.6,
    ))
    ax.add_patch(plt.Circle(
        (world.agent_pos[0], world.agent_pos[1]),
        float(runtime["agent_radius"]), color="blue", alpha=0.9,
    ))
    plt.draw()
    plt.pause(float(runtime["render_pause_seconds"]))


def run_training_and_visualization(
    config_path: str | None = None,
    episodes: int | None = None,
    render: bool | None = None,
    checkpoint_every: int | None = None,
):
    """Treina com brain_config como fonte de parâmetros; argumentos são overrides."""
    config = load_brain_config(config_path)
    runtime = config["runtime"]
    episode_count = int(episodes if episodes is not None else runtime["episodes"])
    should_render = bool(render if render is not None else runtime["render"])
    checkpoint_interval = int(
        checkpoint_every if checkpoint_every is not None
        else runtime.get("checkpoint_every", 0)
    )
    if episode_count <= 0:
        raise ValueError("episodes precisa ser positivo")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_every não pode ser negativo")

    seed = runtime.get("seed")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    world = VirtualWorld2D(config=config)
    brain_engine = SpikeEngine(config=config)
    simulator = TemporalSimulator(engine=brain_engine, config=config)
    memory = MemoryManager(engine=brain_engine, config=config)
    encoder = SensorEncoder(config=config)
    agent = ContinualLearningAgent(simulator, memory, encoder, config=config)

    max_steps = int(runtime["max_steps_per_episode"])
    render_last = int(runtime["render_last_episodes"])
    log_interval = int(runtime["log_interval"])
    window = int(runtime["moving_average_window"])
    history_rewards: list[float] = []

    fig = ax = None
    if should_render:
        plt.ion()
        fig, ax = plt.subplots(figsize=tuple(runtime["render_live_figure_size"]))

    print(
        f"NeuroCore: {episode_count} episódios | passos/episódio={max_steps} | "
        f"config={config['_config_path']}"
    )
    for ep in range(1, episode_count + 1):
        state = world.reset()
        episode_reward = 0.0
        render_live = should_render and ep > episode_count - render_last

        for step in range(1, max_steps + 1):
            action = agent.step(state)
            next_state, reward, done = world.step(action)
            agent.process_feedback(state, action, reward)
            episode_reward += float(reward)
            state = next_state

            if render_live:
                render_world(world, ep, step, ax, runtime)
            if done:
                break

        history_rewards.append(episode_reward)
        if checkpoint_interval and ep % checkpoint_interval == 0:
            memory.consolidate_memory()
            print(f"Checkpoint periódico salvo no episódio {ep}.")
        if ep % log_interval == 0:
            avg_reward = float(np.mean(history_rewards[-log_interval:]))
            print(
                f"Episódio {ep:04d}/{episode_count} | "
                f"recompensa média (últimos {log_interval}): {avg_reward:+.3f}"
            )

    print("\nTreino concluído; consolidando memória...")
    agent.sleep()

    if should_render:
        plt.ioff()
        plt.close(fig)
        plt.figure(figsize=tuple(runtime["final_figure_size"]))
        plt.plot(history_rewards, alpha=0.3, color="blue", label="Recompensa bruta")
        if len(history_rewards) >= window:
            moving_avg = np.convolve(
                history_rewards, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                range(window - 1, len(history_rewards)), moving_avg,
                color="red", linewidth=2, label=f"Média móvel ({window})",
            )
        plt.title("Curva de Aprendizagem do NeuroCore")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa total")
        plt.legend()
        plt.grid(True)
        plt.show()

    return history_rewards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina o NeuroCore com parâmetros de config/brain_config.yaml."
    )
    parser.add_argument("--config", default=None, help="caminho do brain_config.yaml")
    parser.add_argument("--episodes", type=int, default=None, help="override de runtime.episodes")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="salva pesos a cada N episódios")
    parser.add_argument("--no-render", action="store_true", help="desativa renderização e gráfico")
    args = parser.parse_args()
    run_training_and_visualization(
        config_path=args.config,
        episodes=args.episodes,
        render=False if args.no_render else None,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
