"""Teste de retenção em aprendizagem contínua para o classificador LIF.

Protocolo: aprender tarefa A, medir A, aprender tarefa B sem reinicializar
modelo/otimizador e medir A novamente. Compara treino sequencial puro a uma
condição de replay com um exemplar antigo por classe. O conjunto de teste
usa padrões reservados e nunca entra no treino nem no buffer de replay.
"""

from __future__ import annotations

import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from learning.surrogate_gradient import SurrogateLIFClassifier


SEED = 100
REPEATS = 20
EPOCHS_PER_TASK = 120
LEARNING_RATE = 0.01

# Para evitar sobreposição de etiquetas, A contém classes 0/1 e B contém 2/3.
# Cada classe tem três padrões de treino e um par de sensores reservado para teste.
TRAIN_PAIRS = ((0, 1), (0, 2), (1, 2))
TEST_PAIR = (2, 3)
TASK_A_CLASSES = (0, 1)
TASK_B_CLASSES = (2, 3)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pattern(class_id: int, pair: tuple[int, int], device: torch.device) -> torch.Tensor:
    x = torch.zeros(16, device=device)
    offset = class_id * 4
    x[[offset + pair[0], offset + pair[1]]] = 1.0
    return x


def make_task_data(classes: tuple[int, ...], device: torch.device):
    train_x = torch.stack([
        make_pattern(class_id, pair, device)
        for class_id in classes
        for pair in TRAIN_PAIRS
    ])
    train_y = torch.tensor([
        class_id for class_id in classes for _ in TRAIN_PAIRS
    ], dtype=torch.long, device=device)
    test_x = torch.stack([
        make_pattern(class_id, TEST_PAIR, device)
        for class_id in classes
    ])
    test_y = torch.tensor(classes, dtype=torch.long, device=device)
    return train_x, train_y, test_x, test_y


def make_model(device: torch.device) -> SurrogateLIFClassifier:
    return SurrogateLIFClassifier(
        input_size=16,
        hidden_size=128,
        output_size=4,
        steps=12,
    ).to(device)


def train_epochs(
    model: SurrogateLIFClassifier,
    optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int,
    seed: int,
) -> list[float]:
    """Treina sem reinicializar os pesos nem o estado do otimizador."""
    rng = random.Random(seed)
    losses = []
    for _ in range(epochs):
        model.train()
        indices = list(range(train_x.shape[0]))
        rng.shuffle(indices)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=train_x.device)
        optimizer.zero_grad(set_to_none=True)
        spikes, membranes = model(train_x[index_tensor])
        logits = membranes.mean(dim=0)
        spike_counts = spikes.sum(dim=0)
        loss = F.cross_entropy(logits, train_y[index_tensor]) + 0.5 * F.cross_entropy(
            spike_counts, train_y[index_tensor]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses


def evaluate(model: SurrogateLIFClassifier, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        spikes, membranes = model(x)
        counts = spikes.sum(dim=0)
        membrane_logits = membranes.mean(dim=0)
        spike_predictions = counts.argmax(dim=1)
        membrane_predictions = membrane_logits.argmax(dim=1)
        no_spikes = counts.sum(dim=1) == 0
        # An all-zero spike vector has no valid class; do not let argmax
        # silently assign it to class 0.
        spike_predictions[no_spikes] = -1
        return {
            "spike_accuracy": float((spike_predictions == y).float().mean().item()),
            "membrane_accuracy": float((membrane_predictions == y).float().mean().item()),
            "no_spikes": int(no_spikes.sum().item()),
            "predictions": spike_predictions.detach().cpu().tolist(),
            "targets": y.detach().cpu().tolist(),
        }


def run_condition(
    seed: int,
    device: torch.device,
    use_replay: bool,
) -> dict:
    """Roda uma sequência de tarefas em um modelo continuamente carregado."""
    seed_all(seed)
    task_a_train_x, task_a_train_y, task_a_test_x, task_a_test_y = make_task_data(TASK_A_CLASSES, device)
    task_b_train_x, task_b_train_y, task_b_test_x, task_b_test_y = make_task_data(TASK_B_CLASSES, device)

    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Fase A e primeira medição de retenção: desempenho logo após aprender A.
    loss_a = train_epochs(
        model, optimizer, task_a_train_x, task_a_train_y,
        EPOCHS_PER_TASK, seed + 10_000,
    )
    task_a_after_a = evaluate(model, task_a_test_x, task_a_test_y)

    # Buffer limitado: dois padrões de treino de cada classe antiga.
    # Os padrões reservados de teste nunca são usados.
    if use_replay:
        replay_indices = [0, 1, len(TRAIN_PAIRS), len(TRAIN_PAIRS) + 1]
        replay_x = task_a_train_x[replay_indices]
        replay_y = task_a_train_y[replay_indices]
        task_b_train_x = torch.cat([task_b_train_x, replay_x], dim=0)
        task_b_train_y = torch.cat([task_b_train_y, replay_y], dim=0)

    # Fase B: os pesos e o otimizador persistem; não há reset do modelo.
    loss_b = train_epochs(
        model, optimizer, task_b_train_x, task_b_train_y,
        EPOCHS_PER_TASK, seed + 20_000,
    )
    task_a_after_b = evaluate(model, task_a_test_x, task_a_test_y)
    task_b_after_b = evaluate(model, task_b_test_x, task_b_test_y)

    return {
        "seed": seed,
        "condition": "replay_2_per_class" if use_replay else "sequential_no_replay",
        "task_a_after_a": task_a_after_a,
        "task_a_after_b": task_a_after_b,
        "task_b_after_b": task_b_after_b,
        "forgetting_spike_accuracy_pp": 100.0 * (
            task_a_after_a["spike_accuracy"] - task_a_after_b["spike_accuracy"]
        ),
        "loss_a_initial": loss_a[0],
        "loss_a_final": loss_a[-1],
        "loss_b_initial": loss_b[0],
        "loss_b_final": loss_b[-1],
    }


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * sd / (len(values) ** 0.5)
    return mean, mean - margin, mean + margin


def summarize_condition(records: list[dict], condition: str) -> dict:
    selected = [r for r in records if r["condition"] == condition]
    metrics = {
        "task_a_after_a_pct": [100 * r["task_a_after_a"]["spike_accuracy"] for r in selected],
        "task_a_after_b_pct": [100 * r["task_a_after_b"]["spike_accuracy"] for r in selected],
        "task_b_after_b_pct": [100 * r["task_b_after_b"]["spike_accuracy"] for r in selected],
        "forgetting_pp": [r["forgetting_spike_accuracy_pp"] for r in selected],
        "task_a_membrane_after_b_pct": [100 * r["task_a_after_b"]["membrane_accuracy"] for r in selected],
        "task_b_membrane_after_b_pct": [100 * r["task_b_after_b"]["membrane_accuracy"] for r in selected],
    }
    return {
        key: {
            "mean": mean_ci95(values)[0],
            "ci95_low": mean_ci95(values)[1],
            "ci95_high": mean_ci95(values)[2],
        }
        for key, values in metrics.items()
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("APRENDIZAGEM CONTÍNUA — RETENÇÃO TASK A → TASK B")
    print(f"device={device}; seeds={REPEATS}; épocas/tarefa={EPOCHS_PER_TASK}")
    print("Métrica primária: acurácia por contagem de spikes nos testes reservados")
    print("=" * 76)

    records = []
    for repetition in range(REPEATS):
        seed = SEED + repetition
        for replay in (False, True):
            result = run_condition(seed, device, replay)
            records.append(result)
            condition = result["condition"]
            a1 = result["task_a_after_a"]["spike_accuracy"] * 100
            a2 = result["task_a_after_b"]["spike_accuracy"] * 100
            b2 = result["task_b_after_b"]["spike_accuracy"] * 100
            print(
                f"seed={seed} | {condition:22s} | A após A={a1:5.1f}% | "
                f"A após B={a2:5.1f}% | B após B={b2:5.1f}% | "
                f"esquecimento={result['forgetting_spike_accuracy_pp']:+5.1f} pp | "
                f"sem spikes A/B={result['task_a_after_b']['no_spikes']}/"
                f"{result['task_b_after_b']['no_spikes']}"
            )

    summaries = {
        "sequential_no_replay": summarize_condition(records, "sequential_no_replay"),
        "replay_2_per_class": summarize_condition(records, "replay_2_per_class"),
    }
    print("\nRESUMO — média e IC normal aproximado de 95% (unidade: %) ")
    for condition, summary in summaries.items():
        print(f"\n{condition}")
        for metric, values in summary.items():
            print(
                f"  {metric}: {values['mean']:.1f} "
                f"[{values['ci95_low']:.1f}, {values['ci95_high']:.1f}]"
            )

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "task_order": ["A", "B"],
            "task_a_classes": list(TASK_A_CLASSES),
            "task_b_classes": list(TASK_B_CLASSES),
            "repeats": REPEATS,
            "epochs_per_task": EPOCHS_PER_TASK,
            "train_pairs": [list(pair) for pair in TRAIN_PAIRS],
            "held_out_test_pair": list(TEST_PAIR),
            "replay_examples_per_old_class": 2,
            "test_examples_are_never_used_for_training": True,
        },
        "summaries": summaries,
        "runs": records,
        "limitations": [
            "Synthetic class codes only; results do not establish retention on real NeuroCore experiences.",
            "Only two sequential tasks are tested; longer task streams and interference patterns require further tests.",
            "Replay uses one training exemplar per old class and is a comparator, not a complete consolidation mechanism.",
        ],
    }
    output_path = Path(__file__).resolve().parent / "results" / "continual_retention_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResultados completos salvos em: {output_path}")
    print("Interpretação: comparar A após A com A após B; retenção sem esquecimento não é inferida apenas de B após B.")


if __name__ == "__main__":
    main()
