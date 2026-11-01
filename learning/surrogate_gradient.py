"""Surrogate-gradient LIF classifier components.

The forward spike remains a hard threshold. During backpropagation, its
zero/undefined derivative is replaced with the derivative of a fast sigmoid.
This enables supervised BPTT through the finite time-unrolled simulation.
"""

import torch
import torch.nn as nn


class FastSigmoidSpike(torch.autograd.Function):
    """Hard threshold forward pass with fast-sigmoid surrogate derivative."""

    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor, slope: float):
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.slope = float(slope)
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (distance,) = ctx.saved_tensors
        slope = ctx.slope
        surrogate_grad = 1.0 / (1.0 + slope * distance.abs()).square()
        return grad_output * surrogate_grad, None


class SurrogateLIFClassifier(nn.Module):
    """Two-layer rate-coded LIF classifier trained by supervised BPTT.

    The loss can use time-averaged output membrane potentials; inference
    returns output spike counts. Neuron states are local to each call so
    independent samples cannot leak membrane state into one another.
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 128,
        output_size: int = 4,
        steps: int = 12,
        beta: float = 0.99,
        threshold: float = 0.5,
        slope: float = 10.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.steps = steps
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

        # Moderate, reproducible starting scale for sparse binary inputs.
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.2)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        batch_size = inputs.shape[0]
        hidden_mem = inputs.new_zeros((batch_size, self.fc1.out_features))
        output_mem = inputs.new_zeros((batch_size, self.fc2.out_features))
        spike_records = []
        membrane_records = []

        for _ in range(self.steps):
            hidden_mem = self.beta * hidden_mem + self.fc1(inputs)
            hidden_spikes = FastSigmoidSpike.apply(
                hidden_mem - self.threshold, self.slope
            )
            hidden_mem = hidden_mem * (1.0 - hidden_spikes.detach())

            output_mem = self.beta * output_mem + self.fc2(hidden_spikes)
            output_spikes = FastSigmoidSpike.apply(
                output_mem - self.threshold, self.slope
            )
            spike_records.append(output_spikes)
            membrane_records.append(output_mem)
            output_mem = output_mem * (1.0 - output_spikes.detach())

        return torch.stack(spike_records), torch.stack(membrane_records)
