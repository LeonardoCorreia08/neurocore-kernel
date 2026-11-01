"""Valida classificação supervisionada com surrogate-gradient LIF BPTT."""

import random
import statistics

import torch
import torch.nn.functional as F

from learning.surrogate_gradient import SurrogateLIFClassifier

SEED = 42
REPEATS = 20
EPOCHS = 120
LEARNING_RATE = 0.01

TRAIN_PAIRS = ((0, 1), (0, 2), (1, 2))
TEST_PAIR = (2, 3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pattern(class_id: int, pair: tuple[int, int], device: torch.device) -> torch.Tensor:
    x = torch.zeros(16, device=device)
    start = class_id * 4
    x[[start + pair[0], start + pair[1]]] = 1.0
    return x


def datasets(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.stack([
        make_pattern(c, pair, device)
        for c in range(4) for pair in TRAIN_PAIRS
    ])
    train_y = torch.tensor([c for c in range(4) for _ in TRAIN_PAIRS], device=device)
    test_x = torch.stack([make_pattern(c, TEST_PAIR, device) for c in range(4)])
    test_y = torch.arange(4, device=device)
    return train_x, train_y, test_x, test_y


def train_and_evaluate(seed: int, device: torch.device) -> dict:
    set_seed(seed)
    train_x, train_y, test_x, test_y = datasets(device)
    model = SurrogateLIFClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(train_x.shape[0], device=device)
        optimizer.zero_grad(set_to_none=True)
        spikes, membranes = model(train_x[order])
        logits = membranes.mean(dim=0)
        loss = F.cross_entropy(logits, train_y[order])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_spikes, test_membranes = model(test_x)
        spike_counts = test_spikes.sum(dim=0)
        spike_predictions = spike_counts.argmax(dim=1)
        no_spikes_mask = spike_counts.sum(dim=1) == 0
        spike_predictions[no_spikes_mask] = -1
        membrane_predictions = test_membranes.mean(dim=0).argmax(dim=1)
        test_loss = F.cross_entropy(test_membranes.mean(dim=0), test_y).item()
        no_spikes = int(no_spikes_mask.sum().item())
        return {
            "spike_accuracy": (spike_predictions == test_y).float().mean().item(),
            "membrane_accuracy": (membrane_predictions == test_y).float().mean().item(),
            "no_spikes": no_spikes,
            "test_loss": test_loss,
        }


def ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * sd / (len(values) ** 0.5)
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Surrogate-gradient BPTT | device={device} | repeats={REPEATS} | epochs={EPOCHS}")
    spike_scores = []
    membrane_scores = []
    no_spikes_total = 0
    test_losses = []

    for i in range(REPEATS):
        result = train_and_evaluate(SEED + i, device)
        spike_scores.append(result["spike_accuracy"])
        membrane_scores.append(result["membrane_accuracy"])
        no_spikes_total += result["no_spikes"]
        test_losses.append(result["test_loss"])
        print(
            f"seed={SEED + i}: spike-acc={result['spike_accuracy'] * 100:.1f}%, "
            f"membrane-acc={result['membrane_accuracy'] * 100:.1f}%, "
            f"no-spike cases={result['no_spikes']}/4, loss={result['test_loss']:.4f}"
        )

    for label, scores in (("Spike-count accuracy", spike_scores), ("Membrane-logit accuracy", membrane_scores)):
        mean, low, high = ci95(scores)
        print(f"{label}: {mean * 100:.1f}% (approx. 95% CI {low * 100:.1f}–{high * 100:.1f}%)")
    print(f"Total no-spike test cases: {no_spikes_total}/{REPEATS * 4}")
    print(f"Mean test cross-entropy: {statistics.mean(test_losses):.4f}; chance level: 25%")
    print("Held-out examples vary the sensor pair within each class; broader generalization requires new datasets.")


if __name__ == "__main__":
    main()
