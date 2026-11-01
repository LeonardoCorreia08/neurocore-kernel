"""Smoke test: duas sessões devem retomar e estender o treino salvo."""

import json
import os
import tempfile
from pathlib import Path

import torch

from main import run_training_and_visualization


def run_tests() -> None:
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="neurocore-resume-test-") as directory:
        try:
            os.chdir(directory)
            first = run_training_and_visualization(
                episodes=2,
                checkpoint_every=1,
                seed=123,
                checkpoint_name="resume_smoke.pt",
                fresh=True,
                render=False,
            )
            assert first["completed_episodes"] == 2

            first_snapshot = torch.load(
                Path("checkpoints/resume_smoke.pt"),
                map_location="cpu",
                weights_only=True,
            )
            first_count = len(first_snapshot["episodes"])
            assert first_snapshot["runtime_metadata"]["training_state"]["completed_episodes"] == 2

            second = run_training_and_visualization(
                episodes=2,
                checkpoint_every=1,
                seed=123,
                checkpoint_name="resume_smoke.pt",
                fresh=False,
                render=False,
            )
            assert second["completed_episodes"] == 4

            snapshot = torch.load(
                Path("checkpoints/resume_smoke.pt"),
                map_location="cpu",
                weights_only=True,
            )
            training_state = snapshot["runtime_metadata"]["training_state"]
            assert training_state["completed_episodes"] == 4
            assert len(training_state["history_rewards"]) == 4
            assert len(snapshot["episodes"]) >= first_count
            log = json.loads(Path("logs/reinforcement_learning_run.json").read_text())
            assert log["completed_episodes"] == 4
            assert len(log["rewards"]) == 4
        finally:
            os.chdir(original_cwd)

    print("Teste de retomada do treinamento: PASSOU")


if __name__ == "__main__":
    run_tests()
