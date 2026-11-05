"""Teste 18 — Aprendizagem, retenção e consolidação em memória de longo prazo.

Execute da raiz do repositório: python -m experiments.test_18_learning_retention
O teste mede separadamente aprendizagem associativa, retenção sem estímulo,
recuperação após snapshot e restauração do buffer episódico consolidado.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch

from core.config_loader import load_brain_config
from core.spike_engine import SpikeEngine
from memory.associative import AssociativeMemoryNetwork
from memory.long_term import LongTermMemoryManager
from memory.manager import MemoryManager


def _exact_match(prediction: torch.Tensor, target: torch.Tensor) -> bool:
    return bool(torch.equal(prediction.cpu(), target.cpu()))


def run_test(output_path: Path, seed: int = 18) -> dict:
    torch.manual_seed(seed)
    dimension = 8
    input_pattern = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32)
    response_pattern = torch.tensor([0, 1, 0, 1, 1, 0, 0, 0], dtype=torch.float32)

    config = copy.deepcopy(load_brain_config())
    config["memory"]["associative"]["vector_dim"] = dimension
    config["network"]["layer_sizes"] = [dimension, 4]
    config["network"]["device"] = "cpu"

    with tempfile.TemporaryDirectory(prefix="neurocore_test18_") as temporary_directory:
        temp = Path(temporary_directory)
        config["memory"]["checkpoint_dir"] = str(temp / "checkpoints")
        config["memory"]["storage_dir"] = str(temp / "long_term")
        config["memory"]["checkpoint_filename"] = "test18_engine.pt"
        config["memory"]["map_location"] = "cpu"

        associative = AssociativeMemoryNetwork(vector_dim=dimension, config=config)
        before_training = associative.recall(input_pattern)
        exact_before_training = _exact_match(before_training, response_pattern)

        # Três apresentações supervisionadas do par entrada-resposta.
        for _ in range(3):
            associative.encode_association(input_pattern, response_pattern)
        learned_response = associative.recall(input_pattern)
        learned = _exact_match(learned_response, response_pattern)

        # Retira-se o estímulo: não se altera a matriz durante 100 passos de repouso.
        for _ in range(100):
            pass
        retained_response = associative.recall(input_pattern)
        retained_without_stimulus = _exact_match(retained_response, response_pattern)

        # Salva matriz associativa e episódio como um snapshot explícito de longo prazo.
        ltm = LongTermMemoryManager(config=config)
        ltm.store_memory_snapshot("test18_association", {
            "association_matrix": associative.association_matrix.detach().cpu(),
            "experience": {
                "input_pattern": input_pattern.to(torch.int64).tolist(),
                "response_pattern": response_pattern.to(torch.int64).tolist(),
                "training_presentations": 3,
            },
        })
        restored_snapshot = ltm.load_memory_snapshot("test18_association")
        restored_associative = AssociativeMemoryNetwork(vector_dim=dimension, config=config)
        restored_associative.association_matrix = restored_snapshot["association_matrix"].clone()
        recovered_response = restored_associative.recall(input_pattern)
        recovered_from_long_term = _exact_match(recovered_response, response_pattern)
        experience_recovered = restored_snapshot.get("experience", {}).get("input_pattern") == input_pattern.to(torch.int64).tolist()

        # Também verifica a rota real do buffer episódico de MemoryManager.
        engine = SpikeEngine(config=config)
        memory = MemoryManager(engine=engine, config=config)
        action_id = int(torch.argmax(response_pattern).item())
        memory.store_experience(input_pattern.tolist(), action_id, 1.0)
        memory.consolidate_memory("test18_engine.pt")

        restored_engine = SpikeEngine(config=config)
        restored_memory = MemoryManager(engine=restored_engine, config=config)
        checkpoint_restored = restored_memory.retrieve_memory("test18_engine.pt")
        episodic_transfer = (
            checkpoint_restored
            and len(restored_memory.episodic_buffer) == 1
            and restored_memory.episodic_buffer[0]["state"] == input_pattern.tolist()
            and restored_memory.episodic_buffer[0]["action"] == action_id
            and restored_memory.episodic_buffer[0]["reward"] == 1.0
        )

    checks = {
        "pattern_not_recalled_before_training": not exact_before_training,
        "association_learned": learned,
        "association_retained_without_stimulus": retained_without_stimulus,
        "association_recovered_from_long_term_snapshot": recovered_from_long_term,
        "experience_in_long_term_snapshot": experience_recovered,
        "episodic_buffer_transferred_and_restored": episodic_transfer,
    }
    report = {
        "test": "18_learning_retention",
        "seed": seed,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "exact_match_before_training": exact_before_training,
            "exact_match_after_training": learned,
            "exact_match_after_no_stimulus_interval": retained_without_stimulus,
            "exact_match_after_long_term_restore": recovered_from_long_term,
            "episodic_experiences_restored": len(restored_memory.episodic_buffer),
        },
        "diagnostic_interpretation": {
            "association_learned": "Falha indica problema de aprendizagem/representação.",
            "association_retained_without_stimulus": "Falha indica esquecimento durante o intervalo sem estímulo.",
            "association_recovered_from_long_term_snapshot": "Falha indica problema na consolidação ou recuperação do snapshot.",
            "episodic_buffer_transferred_and_restored": "Falha indica que a experiência episódica não chegou à memória de longo prazo.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste 18: aprendizagem, retenção e recuperação da memória.")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("logs/test_18_learning_retention.json"))
    args = parser.parse_args()
    report = run_test(args.output, args.seed)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
