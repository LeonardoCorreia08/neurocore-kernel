
"""Motor central da Spiking Neural Network do NeuroCore."""

import torch
import torch.nn as nn

from core.config_loader import get_config_value

from core.neuron import LeakyIntegrateAndFire
from core.sparse_network import SparseSynapseNetwork
from learning.stdp import STDPManager
from learning.reward_learning import RSTDPManager
from learning.homeostasis import HomeostasisManager


class SpikeEngine(nn.Module):
    def __init__(
        self,
        layer_sizes: list[int] | None = None,
        density: float = 0.2,
        device: str = None,
        enable_stdp: bool = True,
        enable_rstdp: bool = True,
        enable_homeostasis: bool = True,
        config: dict | None = None,
    ):
        super().__init__()

        layer_sizes = get_config_value(config, "network.layer_sizes", layer_sizes)
        if layer_sizes is None:
            raise ValueError("layer_sizes deve ser informado ou definido em config.")
        density = get_config_value(config, "network.connectivity_density", density)
        configured_device = get_config_value(config, "network.device", device)
        device = configured_device if configured_device not in (None, "auto") else None
        enable_stdp = get_config_value(config, "learning.enable_stdp", enable_stdp)
        enable_rstdp = get_config_value(config, "learning.enable_rstdp", enable_rstdp)
        enable_homeostasis = get_config_value(config, "learning.enable_homeostasis", enable_homeostasis)
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes precisa conter pelo menos duas camadas."
            )

        if any(size <= 0 for size in layer_sizes):
            raise ValueError("Todos os tamanhos de camada devem ser positivos.")

        if not 0.0 < density <= 1.0:
            raise ValueError("density precisa estar no intervalo (0, 1].")

        self.config = config
        self.device = torch.device(
            device if device is not None else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.enable_stdp = enable_stdp
        self.enable_rstdp = enable_rstdp
        self.enable_homeostasis = enable_homeostasis
        self.num_layers = len(layer_sizes)

        print(
            f"[NeuroCore Hardware] Motor inicializado no dispositivo: "
            f"{self.device}"
        )

        self.neurons = nn.ModuleList([
            LeakyIntegrateAndFire(
                num_neurons=size,
                threshold=get_config_value(config, "neuron.threshold", 1.0),
                decay=get_config_value(config, "neuron.decay", 0.9),
                config=config,
            )
            for size in layer_sizes
        ]).to(self.device)

        self.synapses = nn.ModuleList()
        self.stdp_managers = nn.ModuleList()
        self.rstdp_managers = nn.ModuleList()
        self.homeostasis_managers = nn.ModuleList()

        for i in range(self.num_layers - 1):
            num_pre = layer_sizes[i]
            num_post = layer_sizes[i + 1]

            self.synapses.append(
                SparseSynapseNetwork(
                    num_post,
                    num_pre,
                    density=density,
                    weight_scale=get_config_value(config, "network.weight_init_scale", 0.1),
                    config=config,
                ).to(self.device)
            )

            self.stdp_managers.append(
                STDPManager(num_post, num_pre, lr=get_config_value(config, "learning.stdp.lr", 0.005), tau_trace=get_config_value(config, "learning.stdp.tau_trace", 20.0)).to(self.device)
            )

            self.rstdp_managers.append(
                RSTDPManager(num_post, num_pre, lr=get_config_value(config, "learning.rstdp.lr", 0.01), tau_trace=get_config_value(config, "learning.rstdp.tau_trace", 20.0), tau_eligibility=get_config_value(config, "learning.rstdp.tau_eligibility", 50.0)).to(self.device)
            )

            self.stdp_managers[-1].A_plus = get_config_value(config, "learning.stdp.a_plus", 1.0)
            self.stdp_managers[-1].A_minus = get_config_value(config, "learning.stdp.a_minus", -1.0)
            self.rstdp_managers[-1].A_plus = get_config_value(config, "learning.rstdp.a_plus", 1.0)
            self.rstdp_managers[-1].A_minus = get_config_value(config, "learning.rstdp.a_minus", -1.0)

            self.homeostasis_managers.append(
                HomeostasisManager(
                    num_post,
                    target_firing_rate=get_config_value(config, "learning.homeostasis.target_firing_rate", 0.1),
                    tau_avg=get_config_value(config, "learning.homeostasis.tau_avg", 100.0),
                    adjustment_rate=get_config_value(config, "learning.homeostasis.adjustment_rate", 0.01),
                    threshold_min=get_config_value(config, "learning.homeostasis.threshold_min", 0.1),
                    threshold_max=get_config_value(config, "learning.homeostasis.threshold_max", 5.0),
                    config=config,
                )
                .to(self.device)
            )

    def _get_indices(self, layer_idx: int) -> torch.Tensor:
        return self.synapses[layer_idx].sparse_weights.indices()

    def forward(self, sensory_current: torch.Tensor) -> list[torch.Tensor]:
        if sensory_current.device != self.device:
            sensory_current = sensory_current.to(self.device)

        spikes = []

        # Camada sensorial
        spk, _ = self.neurons[0](sensory_current)
        spk = spk.view(-1)
        spikes.append(spk)

        # Propagação pelas camadas seguintes
        for i in range(self.num_layers - 1):
            spk_pre = spikes[i]

            current_post = self.synapses[i].forward_propagation(spk_pre)
            spk_post, _ = self.neurons[i + 1](current_post)
            spk_post = spk_post.view(-1)
            spikes.append(spk_post)

            indices = self._get_indices(i)

            if self.enable_stdp:
                deltas_stdp = self.stdp_managers[i].compute_sparse_deltas(
                    spk_pre,
                    spk_post,
                    indices,
                )
                self.synapses[i].update_sparse_weights(deltas_stdp)

            if self.enable_rstdp:
                self.rstdp_managers[i].step_eligibility(
                    spk_pre,
                    spk_post,
                    indices,
                )

            if self.enable_homeostasis:
                self.homeostasis_managers[i].apply_homeostasis(
                    spk_post,
                    self.neurons[i + 1],
                )
                threshold = self.neurons[i + 1].threshold
                if isinstance(threshold, torch.Tensor):
                    self.neurons[i + 1].threshold = threshold.clamp(
                        min=get_config_value(self.config, "learning.homeostasis.threshold_min", get_config_value(self.config, "neuron.threshold_min", 0.1)),
                        max=get_config_value(self.config, "learning.homeostasis.threshold_max", get_config_value(self.config, "neuron.threshold_max", 5.0)),
                    )

        return spikes

    def apply_global_reward(self, reward: float):
        """Aplica recompensa aos rastros de elegibilidade existentes."""
        if not self.enable_rstdp:
            return

        for i in range(self.num_layers - 1):
            indices = self._get_indices(i)

            deltas = self.rstdp_managers[i].apply_reward_modulation(
                reward,
                indices,
            )

            self.synapses[i].update_sparse_weights(deltas)

    def reset_neuron_states(self):
        """Reinicia apenas os estados das membranas neuronais."""
        for neuron in self.neurons:
            neuron.reset_state()

    def reset_learning_states(self):
        """Reinicia os rastros de aprendizagem e as médias homeostáticas."""
        for stdp in self.stdp_managers:
            stdp.reset_traces()

        for rstdp in self.rstdp_managers:
            rstdp.reset_state()

        for homeo in self.homeostasis_managers:
            homeo.reset_state()

    def reset_all_states(self):
        """Reinicia os estados neuronais e os estados de aprendizagem."""
        self.reset_neuron_states()
        self.reset_learning_states()
