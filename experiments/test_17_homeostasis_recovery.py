"""Teste 17 — Homeostase e recuperação neural.

Execute da raiz do repositório: python -m experiments.test_17_homeostasis_recovery
O teste usa uma camada LIF isolada para medir diretamente a homeostase; não
representa um circuito completo com populações excitatórias/inibitórias.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from core.config_loader import load_brain_config
from core.neuron import LeakyIntegrateAndFire
from learning.homeostasis import HomeostasisManager


def _run_phase(lif, homeostasis, current: torch.Tensor, steps: int) -> list[float]:
    rates = []
    for _ in range(steps):
        spikes, _ = lif(current)
        homeostasis.apply_homeostasis(spikes, lif)
        rates.append(float(spikes.float().mean().item()))
    return rates


def run_test(output_path: Path, seed: int = 17) -> dict:
    torch.manual_seed(seed)
    config = copy.deepcopy(load_brain_config())
    config["neuron"].update({"threshold": 1.0, "decay": 0.9, "rest_potential": 0.0})
    config["learning"]["homeostasis"].update({
        "target_firing_rate": 0.10,
        "tau_avg": 20.0,
        "adjustment_rate": 0.05,
        "threshold_min": 0.5,
        "threshold_max": 5.0,
    })

    lif = LeakyIntegrateAndFire(num_neurons=1, config=config)
    homeostasis = HomeostasisManager(num_neurons=1, config=config)
    quiet = torch.zeros(1, dtype=torch.float32)
    continuous_stimulus = torch.tensor([0.65], dtype=torch.float32)

    rest_rates = _run_phase(lif, homeostasis, quiet, 100)
    threshold_before_stimulus = float(lif.threshold.max().item())
    stimulus_rates = _run_phase(lif, homeostasis, continuous_stimulus, 400)
    threshold_after_stimulus = float(lif.threshold.max().item())
    recovery_rates = _run_phase(lif, homeostasis, quiet, 300)

    initial_stimulus_rate = sum(stimulus_rates[:40]) / 40
    late_stimulus_rate = sum(stimulus_rates[-40:]) / 40
    recovered_rate = sum(recovery_rates[-50:]) / 50
    stable_rate = sum(recovery_rates[-100:]) / 100
    threshold_min = float(lif.threshold.min().item())
    threshold_max = float(lif.threshold.max().item())

    # Os módulos atuais têm sinapses de peso assinado, mas não definem tipos de
    # neurônio/populações E e I nem medem balanço excitação-inibição explícito.
    ei_status = "not_implemented_in_tested_architecture"
    checks = {
        "quiet_activity_low": sum(rest_rates) / len(rest_rates) <= 0.05,
        "homeostasis_reduces_excess_activity": late_stimulus_rate < initial_stimulus_rate,
        "threshold_increases_during_overactivity": threshold_after_stimulus > threshold_before_stimulus,
        "activity_recovers_after_stimulus_removed": recovered_rate <= 0.05,
        "post_recovery_activity_is_stable": stable_rate <= 0.05,
        "threshold_stays_within_configured_bounds": 0.5 <= threshold_min <= threshold_max <= 5.0,
    }
    report = {
        "test": "17_homeostasis_recovery",
        "seed": seed,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "rest_firing_rate": sum(rest_rates) / len(rest_rates),
            "stimulus_rate_first_40_steps": initial_stimulus_rate,
            "stimulus_rate_last_40_steps": late_stimulus_rate,
            "recovery_rate_last_50_steps": recovered_rate,
            "stable_rate_last_100_steps": stable_rate,
            "threshold_before_stimulus": threshold_before_stimulus,
            "threshold_after_stimulus": threshold_after_stimulus,
            "threshold_after_recovery": float(lif.threshold.max().item()),
            "threshold_bounds_observed": [threshold_min, threshold_max],
            "excitatory_inhibitory_balance": ei_status,
        },
        "limitations": [
            "Este teste exercita LIF + HomeostasisManager, não a propagação em uma rede multicamada.",
            "A arquitetura disponível não declara populações E/I explícitas; essa métrica não foi avaliada.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste 17: homeostase, retirada do estímulo e estabilidade.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("logs/test_17_homeostasis_recovery.json"))
    args = parser.parse_args()
    report = run_test(args.output, args.seed)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
