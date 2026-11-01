"""Experimento supervisionado de classificação para ablação de plasticidade.

Quatro classes são codificadas em grupos distintos de entradas. Para cada
classe, três padrões de pares ativos são usados no treino e um padrão é
reservado para teste. As quatro condições são comparadas com sementes pareadas.
"""

import random
import statistics

import torch

from core.spike_engine import SpikeEngine


SEED = 42
LAYER_SIZES = [16, 128, 4]
NETWORK_DENSITY = 0.3
WEIGHT_SCALE = 0.5
TRAINING_EPOCHS = 60
TRAINING_STEPS_PER_PATTERN = 5
EVALUATION_STEPS = 5
REPEATS = 20
REWARD_MAGNITUDE = 1.0

# Cada classe ocupa um grupo de 4 sensores. Pares de sensores distintos
# produzem exemplos variados da mesma classe.
TRAIN_PAIRS = ((0, 1), (0, 2), (1, 2))
TEST_PAIR = (2, 3)

CONDITIONS = (
    ("Sem aprendizagem", False, False, False),
    ("STDP", True, False, False),
    ("STDP + R-STDP", True, True, False),
    ("STDP + homeostase", True, False, True),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pattern(class_id: int, pair: tuple[int, int], device: torch.device) -> torch.Tensor:
    pattern = torch.zeros(LAYER_SIZES[0], device=device)
    offset = class_id * 4
    pattern[[offset + pair[0], offset + pair[1]]] = 4.0
    return pattern


def make_datasets(device: torch.device) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    train = [
        (make_pattern(class_id, pair, device), class_id)
        for class_id in range(4)
        for pair in TRAIN_PAIRS
    ]
    test = [
        (make_pattern(class_id, TEST_PAIR, device), class_id)
        for class_id in range(4)
    ]
    return train, test


def create_engine(device: torch.device, stdp: bool, rstdp: bool, homeostasis: bool) -> SpikeEngine:
    return SpikeEngine(
        layer_sizes=LAYER_SIZES,
        density=NETWORK_DENSITY,
        device=str(device),
        enable_stdp=stdp,
        enable_rstdp=rstdp,
        enable_homeostasis=homeostasis,
        weight_scale=WEIGHT_SCALE,
    )


def inference_counts(engine: SpikeEngine, pattern: torch.Tensor, steps: int = EVALUATION_STEPS) -> torch.Tensor:
    """Inferência sem plasticidade, homeostase ou alteração dos pesos."""
    engine.reset_neuron_states()
    old_flags = (engine.enable_stdp, engine.enable_rstdp, engine.enable_homeostasis)
    engine.enable_stdp = engine.enable_rstdp = engine.enable_homeostasis = False
    counts = torch.zeros(LAYER_SIZES[-1], dtype=torch.float32)
    try:
        with torch.no_grad():
            for _ in range(steps):
                counts += engine(pattern)[-1].detach().cpu().float()
    finally:
        engine.enable_stdp, engine.enable_rstdp, engine.enable_homeostasis = old_flags
    return counts


def supervised_reward(counts: torch.Tensor, target: int) -> float:
    """Recompensa o alvo relativamente ao maior canal incorreto."""
    total = float(counts.sum().item())
    if total == 0.0:
        return 0.0
    target_count = float(counts[target].item())
    other_counts = counts.clone()
    other_counts[target] = float("-inf")
    strongest_wrong = float(other_counts.max().item())
    margin = (target_count - strongest_wrong) / total
    return max(-REWARD_MAGNITUDE, min(REWARD_MAGNITUDE, margin))


def train_one(engine: SpikeEngine, train_data: list[tuple[torch.Tensor, int]], seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(TRAINING_EPOCHS):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        for index in order:
            pattern, target = train_data[index]
            engine.reset_neuron_states()
            for _ in range(TRAINING_STEPS_PER_PATTERN):
                engine(pattern)
            if engine.enable_rstdp:
                counts = inference_counts(engine, pattern)
                reward = supervised_reward(counts, target)
                engine.apply_global_reward(reward)


def evaluate(engine: SpikeEngine, test_data: list[tuple[torch.Tensor, int]]) -> dict:
    correct = 0
    no_spikes = 0
    confusion = [[0 for _ in range(4)] for _ in range(4)]
    for pattern, target in test_data:
        counts = inference_counts(engine, pattern)
        if float(counts.sum().item()) == 0.0:
            no_spikes += 1
            predicted = -1
        else:
            predicted = int(torch.argmax(counts).item())
        if predicted >= 0:
            confusion[target][predicted] += 1
        if predicted == target:
            correct += 1
    return {
        "accuracy": correct / len(test_data),
        "correct": correct,
        "total": len(test_data),
        "no_spikes": no_spikes,
        "confusion": confusion,
    }


def weight_summary(engine: SpikeEngine) -> list[dict]:
    result = []
    for layer, synapse in enumerate(engine.synapses, start=1):
        values = synapse.sparse_weights.coalesce().values().detach().float().cpu()
        result.append({
            "layer": layer,
            "mean": values.mean().item(),
            "std": values.std(unbiased=False).item(),
            "min": values.min().item(),
            "max": values.max().item(),
            "positive_pct": (values > 0).float().mean().item() * 100,
            "negative_pct": (values < 0).float().mean().item() * 100,
            "clipped_pct": ((values <= -1.0) | (values >= 1.0)).float().mean().item() * 100,
        })
    return result


def summarize(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    # Intervalo normal aproximado de 95%; limitar limites ao intervalo [0, 1].
    half = 1.96 * sd / (len(values) ** 0.5) if values else 0.0
    return mean, max(0.0, mean - half), min(1.0, mean + half)


def run_ablation_study() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("CLASSIFICAÇÃO SUPERVISIONADA — ABLAÇÃO DE PLASTICIDADE")
    print(f"Dispositivo={device}; seed base={SEED}; repetições={REPEATS}; épocas={TRAINING_EPOCHS}")
    print(f"Treino={4 * len(TRAIN_PAIRS)} padrões; teste={4} padrões reservados; escala de peso={WEIGHT_SCALE}")
    print("=" * 78)

    train_data, test_data = make_datasets(device)
    scores: dict[str, list[float]] = {name: [] for name, *_ in CONDITIONS}
    no_spike_totals: dict[str, int] = {name: 0 for name, *_ in CONDITIONS}
    latest_weights: dict[str, list[dict]] = {}

    for repeat in range(REPEATS):
        seed = SEED + repeat
        print(f"\nRepetição {repeat + 1}/{REPEATS} (seed={seed})")
        for name, stdp, rstdp, homeostasis in CONDITIONS:
            # Mesma seed reinicializa topologia/pesos para comparação pareada.
            set_seed(seed)
            engine = create_engine(device, stdp, rstdp, homeostasis)
            if stdp or rstdp or homeostasis:
                train_one(engine, train_data, seed)
            result = evaluate(engine, test_data)
            scores[name].append(result["accuracy"])
            no_spike_totals[name] += result["no_spikes"]
            latest_weights[name] = weight_summary(engine)
            print(
                f"  {name:22s}: acurácia={result['correct']}/{result['total']} "
                f"({result['accuracy'] * 100:.1f}%), sem spikes={result['no_spikes']}"
            )

    print("\n" + "=" * 78 + "\nRESUMO AGREGADO (por repetição; 95% IC aproximado)\n" + "=" * 78)
    for name, *_ in CONDITIONS:
        mean, low, high = summarize(scores[name])
        print(f"{name:22s}: {mean * 100:.1f}% [{low * 100:.1f}, {high * 100:.1f}] | sem spikes nos testes={no_spike_totals[name]}/{REPEATS * len(test_data)}")
        print("  Última repetição — resumo dos pesos:")
        for item in latest_weights[name]:
            print(
                f"    camada {item['layer']}: média={item['mean']:+.4f}, desvio={item['std']:.4f}, "
                f"faixa=[{item['min']:+.3f}, {item['max']:+.3f}], clip={item['clipped_pct']:.1f}%"
            )
    print("\nAcurácia considera cada padrão de teste; ausência de spikes conta como erro.")
    print("O conjunto de teste contém variações reservadas da mesma codificação de classe; não prova generalização para novos domínios.")
    print("=" * 78)


if __name__ == "__main__":
    run_ablation_study()
