"""Treinamento retomável do agente NeuroCore no VirtualWorld2D."""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.simulation import TemporalSimulator
from core.spike_engine import SpikeEngine
from environment.world import VirtualWorld2D
from learning.continual_learning import ContinualLearningAgent
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder

CHECKPOINT_DEFAULT = "latest_brain_engram_directional.pt"
LOG_PATH = Path("logs/reinforcement_learning_run.json")
EXPLORATION_DECAY_EPISODES = 1000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_world(world, episode, step, ax):
    """Desenha o ambiente 2D durante os episódios finais da execução atual."""
    ax.clear()
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_title(f"NeuroCore | Episódio global: {episode} | Passo: {step}")
    for obs in world.obstacles:
        circle = plt.Circle(
            (obs[0], obs[1]), world.obstacle_radius, color="red", alpha=0.6
        )
        ax.add_patch(circle)
    goal = plt.Circle(
        (world.goal_pos[0], world.goal_pos[1]), world.goal_radius,
        color="green", alpha=0.6,
    )
    ax.add_patch(goal)
    agent_patch = plt.Circle(
        (world.agent_pos[0], world.agent_pos[1]), 2.0, color="blue", alpha=0.9
    )
    ax.add_patch(agent_patch)
    plt.draw()
    plt.pause(0.05)


def _training_state(seed, completed, rewards, successes, collisions, silence, actions):
    return {
        "base_seed": int(seed),
        "completed_episodes": int(completed),
        "history_rewards": [float(v) for v in rewards],
        "history_successes": [int(v) for v in successes],
        "history_collisions": [int(v) for v in collisions],
        "history_silence": [int(v) for v in silence],
        "history_actions": [[int(x) for x in row] for row in actions],
    }


def _write_log(payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{LOG_PATH.name}.", suffix=".tmp", dir=LOG_PATH.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(temporary, LOG_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_training_and_visualization(
    episodes: int = 1000,
    checkpoint_every: int = 100,
    seed: int = 42,
    checkpoint_name: str = CHECKPOINT_DEFAULT,
    fresh: bool = False,
    render: bool = True,
) -> dict:
    """Train `episodes` additional episodes, resuming from checkpoint by default."""
    if episodes <= 0:
        raise ValueError("episodes precisa ser positivo")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every precisa ser positivo")

    seed_everything(seed)
    checkpoint_path = Path("checkpoints") / checkpoint_name
    if fresh and checkpoint_path.exists():
        raise FileExistsError(
            f"O checkpoint {checkpoint_path} já existe. Use outro --checkpoint-name "
            "para iniciar um experimento novo sem sobrescrever o aprendizado existente."
        )

    world = VirtualWorld2D(width=100, height=100, sensor_mode="directional")
    brain_engine = SpikeEngine(layer_sizes=[6, 32, 4], density=0.25)
    simulator = TemporalSimulator(engine=brain_engine, time_steps=20)
    memory = MemoryManager(engine=brain_engine, save_dir="checkpoints/")
    encoder = SensorEncoder(max_rate=100.0)
    agent = ContinualLearningAgent(simulator, memory, encoder)

    completed_episodes = 0
    history_rewards: list[float] = []
    history_successes: list[int] = []
    history_collisions: list[int] = []
    history_silence: list[int] = []
    history_actions: list[list[int]] = []

    if not fresh and agent.restore_memory(checkpoint_name):
        saved = memory.restored_runtime.get("training_state", {})
        saved_seed = saved.get("base_seed")
        if saved_seed is not None and int(saved_seed) != seed:
            raise ValueError(
                f"Checkpoint foi treinado com seed {saved_seed}, mas foi pedido seed {seed}. "
                "Use a seed original para continuar ou outro checkpoint para um experimento novo."
            )
        completed_episodes = int(saved.get("completed_episodes", 0))
        history_rewards = [float(v) for v in saved.get("history_rewards", [])]
        history_successes = [int(v) for v in saved.get("history_successes", [])]
        history_collisions = [int(v) for v in saved.get("history_collisions", [])]
        history_silence = [int(v) for v in saved.get("history_silence", [])]
        history_actions = [
            [int(x) for x in row] for row in saved.get("history_actions", [])
        ]
        if "completed_episodes" not in saved and LOG_PATH.is_file():
            try:
                legacy_log = json.loads(LOG_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                legacy_log = {}
            if legacy_log.get("seed") == seed:
                completed_episodes = int(
                    legacy_log.get("completed_episodes", legacy_log.get("episodes", 0))
                )
                history_rewards = [
                    float(v) for v in legacy_log.get("history_rewards", legacy_log.get("rewards", []))
                ]
                history_successes = [
                    int(v) for v in legacy_log.get("history_successes", legacy_log.get("successes", []))
                ]
                history_collisions = [
                    int(v) for v in legacy_log.get("history_collisions", legacy_log.get("collisions", []))
                ]
                history_silence = [
                    int(v) for v in legacy_log.get("history_silence", legacy_log.get("silent_decisions", []))
                ]
                history_actions = [
                    [int(x) for x in row]
                    for row in legacy_log.get("history_actions", legacy_log.get("action_counts_per_episode", []))
                ]
                print(
                    f"Progresso migrado do log antigo: {completed_episodes} episódios."
                )
        print(
            f"Checkpoint retomado: {completed_episodes} episódios concluídos; "
            f"buffer={len(memory.episodic_buffer)} transições."
        )
    elif fresh:
        print("Iniciando treinamento novo; checkpoint existente não será substituído.")
    else:
        print("Nenhum checkpoint encontrado; iniciando treinamento novo.")

    # Cria uma sequência reprodutível específica para a sessão retomada.
    seed_everything(seed + completed_episodes)

    fig = ax = None
    if render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

    epsilon_start, epsilon_end = 1.0, 0.05
    print(
        f"Sessão: +{episodes} episódios; checkpoint a cada {checkpoint_every}; "
        f"epsilon decai nos primeiros {EXPLORATION_DECAY_EPISODES} episódios acumulados."
    )

    def persist_progress(replay_stats=None):
        state = _training_state(
            seed, completed_episodes, history_rewards, history_successes,
            history_collisions, history_silence, history_actions,
        )
        payload = {
            "seed": seed,
            "completed_episodes": completed_episodes,
            "episodes_this_run": episodes,
            "max_steps": 50,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "td_critic_enabled": agent.use_td_critic,
            "rewards": history_rewards,
            "successes": history_successes,
            "collisions": history_collisions,
            "silent_decisions": history_silence,
            "action_counts_per_episode": history_actions,
            "reward_baseline": agent.reward_baseline,
            "checkpoint_name": checkpoint_name,
            "replay": replay_stats,
        }
        return state, payload

    for local_episode in range(1, episodes + 1):
        global_episode = completed_episodes + 1
        simulator.reset_for_new_episode()
        state = world.reset()
        progress = min(
            (global_episode - 1) / max(EXPLORATION_DECAY_EPISODES - 1, 1),
            1.0,
        )
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        episode_reward = 0.0
        episode_actions = np.zeros(4, dtype=np.int64)
        silent_decisions = 0
        success = collision = False
        render_live = render and local_episode > episodes - 5

        for step in range(1, 51):
            action = agent.step(state, epsilon=epsilon)
            episode_actions[action] += 1
            silent_decisions += int(simulator.last_action_silent)
            next_state, reward, done = world.step(action)
            truncated = step == 50 and not done
            agent.process_feedback(
                state,
                action,
                reward,
                next_state=next_state,
                done=done,
                task_id="navigation",
                truncated=truncated,
            )
            episode_reward += reward
            state = next_state
            if render_live:
                render_world(world, global_episode, step, ax)
            if done:
                success = reward > 0.0
                collision = reward < 0.0
                break

        completed_episodes += 1
        history_rewards.append(float(episode_reward))
        history_successes.append(int(success))
        history_collisions.append(int(collision))
        history_silence.append(int(silent_decisions))
        history_actions.append(episode_actions.tolist())

        if completed_episodes % 100 == 0:
            window_start = max(0, len(history_rewards) - 100)
            recent_actions = np.asarray(
                history_actions[window_start:], dtype=np.int64
            ).sum(axis=0).tolist()
            print(
                f"Ep {completed_episodes} | eps={epsilon:.3f} | "
                f"retorno={np.mean(history_rewards[window_start:]):+.3f} | "
                f"sucesso={100*np.mean(history_successes[window_start:]):.1f}% | "
                f"colisão={100*np.mean(history_collisions[window_start:]):.1f}% | "
                f"silêncio/ep={np.mean(history_silence[window_start:]):.1f} | "
                f"ações={recent_actions}"
            )

        if local_episode % checkpoint_every == 0:
            state_snapshot, log_payload = persist_progress()
            path = agent.save_checkpoint(
                checkpoint_name,
                runtime_metadata={"training_state": state_snapshot},
            )
            _write_log(log_payload)
            print(f"Checkpoint periódico salvo: {path}")

    print("\nFinalizando a sessão: replay de consolidação e checkpoint...")
    final_state, _ = persist_progress()
    replay_stats = agent.sleep(
        memory_name=checkpoint_name,
        replay_batch_size=32,
        replay_reward_scale=0.5,
        task_id="navigation",
        runtime_metadata={"training_state": final_state},
    )
    _, final_log = persist_progress(replay_stats)
    _write_log(final_log)

    last_count = min(100, len(history_successes))
    last_success = 100.0 * float(np.mean(history_successes[-last_count:]))
    print(
        f"Replay={replay_stats.get('replayed', 0)}; episódios acumulados="
        f"{completed_episodes}; sucesso últimos {last_count}={last_success:.1f}%; "
        f"checkpoint={checkpoint_path}."
    )

    if render:
        plt.ioff()
        plt.close(fig)
        plt.figure(figsize=(10, 5))
        plt.plot(history_rewards, alpha=0.3, color="blue", label="Retorno por episódio")
        window = 50
        if len(history_rewards) >= window:
            moving_avg = np.convolve(
                history_rewards, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                range(window - 1, len(history_rewards)), moving_avg,
                color="red", linewidth=2, label=f"Média móvel ({window})",
            )
        plt.title("NeuroCore RL: retorno por episódio acumulado")
        plt.xlabel("Episódio global")
        plt.ylabel("Retorno")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "completed_episodes": completed_episodes,
        "episodes_this_run": episodes,
        "success_last_100_percent": last_success,
        "checkpoint": str(checkpoint_path),
        "log": str(LOG_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina o NeuroCore retomando pesos e memória do checkpoint."
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="episódios adicionais nesta execução (padrão: 1000)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="salvar progresso a cada N episódios (padrão: 100)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed do experimento")
    parser.add_argument(
        "--checkpoint-name", default=CHECKPOINT_DEFAULT,
        help="nome relativo do checkpoint dentro de checkpoints/",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="iniciar do zero; recusa sobrescrever um checkpoint já existente",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="desativar visualização para execução headless/mais rápida",
    )
    args = parser.parse_args()
    run_training_and_visualization(
        episodes=args.episodes,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        checkpoint_name=args.checkpoint_name,
        fresh=args.fresh,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
