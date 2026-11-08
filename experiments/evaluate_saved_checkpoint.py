"""Avalia um checkpoint treinado sem atualizar pesos nem guardar experiências."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics

import torch

from core.simulation import TemporalSimulator
from core.spike_engine import SpikeEngine
from experiments.reinforcement_learning_assessment import (
    evaluate_agent,
    evaluate_random,
    fixed_eval_cases,
    seed_all,
)
from learning.continual_learning import ContinualLearningAgent
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder

DEFAULT_CHECKPOINT = "latest_brain_engram_directional.pt"
DEFAULT_MAP_SEEDS = (510, 511, 512, 513, 514)


def evaluate_checkpoint(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    episodes_per_seed: int = 100,
    map_seeds: tuple[int, ...] = DEFAULT_MAP_SEEDS,
    output: str | None = None,
) -> dict:
    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed precisa ser positivo")
    if not map_seeds:
        raise ValueError("Informe ao menos uma seed de mapas")

    seed_all(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = SpikeEngine(
        layer_sizes=[6, 32, 4],
        density=0.25,
        device=str(device),
        enable_stdp=True,
        enable_rstdp=True,
        enable_homeostasis=True,
    )
    simulator = TemporalSimulator(engine, time_steps=20)
    memory = MemoryManager(engine, save_dir="checkpoints/")
    agent = ContinualLearningAgent(
        simulator,
        memory,
        SensorEncoder(max_rate=100.0),
    )
    if not memory._load_checkpoint(checkpoint_name):
        raise FileNotFoundError(
            f"Checkpoint não encontrado: checkpoints/{checkpoint_name}. "
            f"Checkpoint não encontrado: checkpoints/{checkpoint_name}. "
            "Execute o treino antes da avaliação."
        )

    runtime = memory.restored_runtime
    training_state = runtime.get("training_state", {})
    rows = []
    for map_seed in map_seeds:
        cases = fixed_eval_cases(map_seed, episodes_per_seed)
        random_result = evaluate_random(cases, map_seed)
        trained_result = evaluate_agent(agent, cases, map_seed)
        rows.append({
            "map_seed": int(map_seed),
            "episodes": episodes_per_seed,
            "random_policy": random_result,
            "saved_checkpoint": trained_result,
        })
        silence_rate = trained_result["silent_decisions"] / max(
            trained_result["mean_steps"] * episodes_per_seed, 1.0
        )
        print(
            f"map_seed={map_seed}: aleatório="
            f"{random_result['successes']}/{episodes_per_seed}; salvo="
            f"{trained_result['successes']}/{episodes_per_seed}; "
            f"recompensa={trained_result['mean_reward']:+.3f}; "
            f"silêncio motor={100*silence_rate:.1f}%"
        )

    total_episodes = episodes_per_seed * len(map_seeds)
    random_successes = sum(row["random_policy"]["successes"] for row in rows)
    trained_successes = sum(row["saved_checkpoint"]["successes"] for row in rows)
    summary = {
        "episodes": total_episodes,
        "random_success_rate_percent": 100.0 * random_successes / total_episodes,
        "checkpoint_success_rate_percent": 100.0 * trained_successes / total_episodes,
        "random_mean_reward": statistics.mean(
            row["random_policy"]["mean_reward"] for row in rows
        ),
        "checkpoint_mean_reward": statistics.mean(
            row["saved_checkpoint"]["mean_reward"] for row in rows
        ),
        "checkpoint_mean_steps": statistics.mean(
            row["saved_checkpoint"]["mean_steps"] for row in rows
        ),
        "checkpoint_silent_decisions": sum(
            row["saved_checkpoint"]["silent_decisions"] for row in rows
        ),
        "checkpoint_total_decisions": sum(
            row["saved_checkpoint"]["mean_steps"] * episodes_per_seed
            for row in rows
        ),
    }
    summary["checkpoint_silent_rate_percent"] = 100.0 * (
        summary["checkpoint_silent_decisions"]
        / max(summary["checkpoint_total_decisions"], 1.0)
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_name": checkpoint_name,
        "checkpoint_completed_training_episodes": training_state.get(
            "completed_episodes"
        ),
        "device": str(device),
        "protocol": {
            "episodes_per_map_seed": episodes_per_seed,
            "map_seeds": list(map_seeds),
            "plasticity_disabled_during_evaluation": True,
            "checkpoint_weights_updated_during_evaluation": False,
            "notes": [
                "The same held-out maps are used for random and checkpoint policies.",
                "Map seeds vary evaluation cases; they are not independent training seeds.",
                "This test evaluates this saved checkpoint, unlike the fresh-training benchmark.",
            ],
        },
        "summary": summary,
        "runs": rows,
    }
    output_path = Path(output) if output else (
        Path(__file__).resolve().parent
        / "results"
        / f"checkpoint_evaluation_{Path(checkpoint_name).stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nRESUMO DO CHECKPOINT SALVO")
    print(
        f"Treino acumulado: {training_state.get('completed_episodes', 'desconhecido')} episódios; "
        f"sucesso checkpoint={summary['checkpoint_success_rate_percent']:.1f}% "
        f"({trained_successes}/{total_episodes}); aleatório="
        f"{summary['random_success_rate_percent']:.1f}% "
        f"({random_successes}/{total_episodes})."
    )
    print(
        f"Recompensa média checkpoint={summary['checkpoint_mean_reward']:+.3f}; "
        f"silêncio motor={summary['checkpoint_silent_rate_percent']:.1f}%."
    )
    print(f"Resultados detalhados: {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia um checkpoint existente em mapas reservados, sem aprender."
    )
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument(
        "--map-seeds", type=int, nargs="+", default=list(DEFAULT_MAP_SEEDS)
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_name=args.checkpoint_name,
        episodes_per_seed=args.episodes_per_seed,
        map_seeds=tuple(args.map_seeds),
        output=args.output,
    )


if __name__ == "__main__":
    main()
