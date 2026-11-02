'''Modulo: experiment_plotter.py'''
import os
import matplotlib.pyplot as plt
import torch


def _settings(config):
    if not config:
        return {}
    return config.get("monitoring", config).get("experiment_plotter", {})


class ExperimentPlotter:
    def __init__(self, output_dir: str = "experiments/plots", max_neurons: int = 150, dpi: int = 300, filenames=None, config: dict = None):
        settings = _settings(config)
        self.output_dir = settings.get("output_dir", output_dir)
        self.max_neurons = settings.get("max_neurons", max_neurons)
        self.dpi = settings.get("dpi", dpi)
        self.filenames = settings.get("filenames", filenames or ["learning_curve.png", "raster_plot.png", "energy_profile.png"])
        os.makedirs(self.output_dir, exist_ok=True)

    def _filename(self, index, explicit):
        return explicit if explicit is not None else self.filenames[index]

    def plot_learning_curve(self, episode_rewards: list[float], filename: str = None):
        """Gera o gráfico da curva de aprendizado (recompensas ou metas ao longo dos episódios)."""
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, color='tab:blue', linewidth=2, marker='o', markersize=4)
        plt.title("NeuroCore - Curva de Aprendizado e Desempenho", fontsize=12, fontweight='bold')
        plt.xlabel("Episódios", fontsize=10)
        plt.ylabel("Metas / Recompensa Acumulada", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, self._filename(0, filename))
        plt.savefig(filepath, dpi=self.dpi)
        plt.close()
        print(f"[Plotter] Curva de aprendizado salva em: {filepath}")

    def plot_raster(self, spike_history: list[torch.Tensor], filename: str = None, max_neurons: int = None):
        """Gera e salva o gráfico Raster Plot da atividade neural (spikes ao longo do tempo)."""
        if not spike_history:
            print("[Plotter] Nenhum histórico de spikes para plotar.")
            return

        history_matrix = torch.stack(spike_history).detach().cpu().numpy()
        time_steps, num_neurons = history_matrix.shape
        max_neurons = self.max_neurons if max_neurons is None else max_neurons
        count = min(num_neurons, max_neurons)
        plt.figure(figsize=(12, 5))
        for neuron_id in range(count):
            spike_times = (history_matrix[:, neuron_id] > 0).nonzero()[0]
            if len(spike_times) > 0:
                plt.scatter(spike_times, [neuron_id] * len(spike_times), s=3, color='tab:purple', alpha=0.8)
        plt.title(f"NeuroCore - Raster Plot de Atividade Neural (População de {count} neurônios)", fontsize=12, fontweight='bold')
        plt.xlabel("Passos de Tempo (ms)", fontsize=10)
        plt.ylabel("ID do Neurônio", fontsize=10)
        plt.xlim(0, time_steps)
        plt.ylim(-1, count)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, self._filename(1, filename))
        plt.savefig(filepath, dpi=self.dpi)
        plt.close()
        print(f"[Plotter] Raster plot salvo em: {filepath}")

    def plot_energy_and_sops(self, steps: list[int], energy_values: list[float], filename: str = None):
        """Gera gráfico do perfil de consumo energético e custo computacional."""
        plt.figure(figsize=(10, 4))
        plt.plot(steps, energy_values, color='tab:red', linewidth=2, marker='s', markersize=4)
        plt.title("NeuroCore - Perfil de Consumo Energético por Ciclo", fontsize=12, fontweight='bold')
        plt.xlabel("Passos de Simulação", fontsize=10)
        plt.ylabel("Energia Estimada (Joules)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, self._filename(2, filename))
        plt.savefig(filepath, dpi=self.dpi)
        plt.close()
        print(f"[Plotter] Gráfico de energia salvo em: {filepath}")
