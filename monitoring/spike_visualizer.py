'''Modulo: spike_visualizer.py'''
import os
import matplotlib.pyplot as plt
import torch


def _settings(config):
    if not config:
        return {}
    return config.get("monitoring", config).get("spike_visualizer", {})


class SpikeVisualizer:
    def __init__(self, output_file: str = "docs/spike_raster.png", figsize=(10, 4), dpi: int = 300, config: dict = None):
        settings = _settings(config)
        self.output_file = settings.get("output_file", output_file)
        self.figsize = settings.get("figsize", figsize)
        self.dpi = settings.get("dpi", dpi)
        self.spike_history = []

    def record_spikes(self, spike_tensor: torch.Tensor):
        """Registra os spikes de um passo de tempo para posterior visualizacao."""
        self.spike_history.append(spike_tensor.detach().cpu().clone())

    @staticmethod
    def plot_raster(spike_history=None, filename: str = None, title: str = "NeuroCore - Raster Plot de Atividade Neuronal", config: dict = None):
        """Plota histórico como lista ou instância, com configurações opcionais."""
        settings = _settings(config)
        filename = settings.get("output_file", "docs/spike_raster.png") if filename is None else filename
        figsize = settings.get("figsize", [10, 4])
        dpi = settings.get("dpi", 300)
        history = getattr(spike_history, "spike_history", spike_history)
        if history is None:
            history = []
        if not history:
            print("[Visualizer] Nenhum spike registrado para plotagem.")
            return

        spikes_tensor = torch.stack(history).detach().cpu().numpy()
        steps, neurons = spikes_tensor.shape
        plt.figure(figsize=figsize)
        for neuron_id in range(neurons):
            spike_times = [t for t in range(steps) if spikes_tensor[t, neuron_id] > 0]
            if spike_times:
                plt.scatter(spike_times, [neuron_id] * len(spike_times), marker='|', color='black', s=10)
        plt.title(title)
        plt.xlabel("Passos de Tempo (Steps)")
        plt.ylabel("ID do Neurônio")
        plt.xlim(0, steps)
        plt.ylim(-1, neurons)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if filename:
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Raster Plot salvo com sucesso em '{filename}'!")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_static(spike_history: list, filename: str = None, config: dict = None):
        """Método utilitário estático caso queira passar uma lista crua diretamente."""
        SpikeVisualizer.plot_raster(spike_history, filename=filename, config=config)

    def reset(self):
        self.spike_history.clear()
