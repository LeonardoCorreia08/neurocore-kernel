"""Smoke test do avaliador offline de checkpoints persistidos."""

import os
import tempfile
from pathlib import Path

import torch

from core.simulation import TemporalSimulator
from core.spike_engine import SpikeEngine
from experiments.evaluate_saved_checkpoint import evaluate_checkpoint
from learning.continual_learning import ContinualLearningAgent
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder


def run_tests() -> None:
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="neurocore-checkpoint-eval-") as directory:
        try:
            os.chdir(directory)
            torch.manual_seed(9)
            engine = SpikeEngine(
                layer_sizes=[6, 32, 4], density=0.25, device="cpu"
            )
            simulator = TemporalSimulator(engine, time_steps=10)
            memory = MemoryManager(engine, save_dir="checkpoints/")
            agent = ContinualLearningAgent(
                simulator, memory, SensorEncoder(max_rate=100.0)
            )
            memory.save_checkpoint(
                "smoke.pt",
                runtime_metadata={
                    "training_state": {
                        "base_seed": 9,
                        "completed_episodes": 0,
                    }
                },
            )
            result = evaluate_checkpoint(
                checkpoint_name="smoke.pt",
                episodes_per_seed=2,
                map_seeds=(31, 32),
                output="results/eval.json",
            )
            assert result["summary"]["episodes"] == 4
            assert len(result["runs"]) == 2
            assert result["protocol"]["plasticity_disabled_during_evaluation"]
            assert result["protocol"]["checkpoint_weights_updated_during_evaluation"] is False
            assert Path("results/eval.json").is_file()
            assert engine.enable_stdp and engine.enable_rstdp and engine.enable_homeostasis
        finally:
            os.chdir(original_cwd)

    print("Teste de avaliação do checkpoint salvo: PASSOU")


if __name__ == "__main__":
    run_tests()
