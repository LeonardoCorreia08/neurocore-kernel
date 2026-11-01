'''Modulo: experiment_plotter.py'''
import os
import matplotlib.pyplot as plt
import torch

class ExperimentPlotter:
    def __init__(self, output_dir: str = "experiments/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_learning_curve(self, episode_rewards: list[float], filename: str = "learning_curve.png"):
        """Gera o gráfico da curva de aprendizado (recompensas ou metas ao longo dos episódios)."""
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, color='tab:blue', linewidth=2, marker='o', markersize=4)
        plt.title("NeuroCore - Curva de Aprendizado e Desempenho", fontsize=12, fontweight='bold')
        plt.xlabel("Episódios", fontsize=10)
        plt.ylabel("Metas / Recompensa Acumulada", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[Plotter] Curva de aprendizado salva em: {filepath}")

    def plot_raster(self, spike_history: list[torch.Tensor], filename: str = "raster_plot.png", max_neurons: int = 150):
        """Gera e salva o gráfico Raster Plot da atividade neural (spikes ao longo do tempo)."""
        if not spike_history:
            print("[Plotter] Nenhum histórico de spikes para plotar.")
            return

        history_matrix = torch.stack(spike_history).numpy() # [tempo, neurônios]
        time_steps, num_neurons = history_matrix.shape

        plt.figure(figsize=(12, 5))
        for neuron_id in range(min(num_neurons, max_neurons)):
            spike_times = torch.where(torch.tensor(history_matrix[:, neuron_id]) > 0)[0].numpy()
            if len(spike_times) > 0:
                plt.scatter(spike_times, [neuron_id] * len(spike_times), s=3, color='tab:purple', alpha=0.8)

        plt.title(f"NeuroCore - Raster Plot de Atividade Neural (População de {min(num_neurons, max_neurons)} neurônios)", fontsize=12, fontweight='bold')
        plt.xlabel("Passos de Tempo (ms)", fontsize=10)
        plt.ylabel("ID do Neurônio", fontsize=10)
        plt.xlim(0, time_steps)
        plt.ylim(-1, min(num_neurons, max_neurons))
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[Plotter] Raster plot salvo em: {filepath}")

    def plot_energy_and_sops(self, steps: list[int], energy_values: list[float], filename: str = "energy_profile.png"):
        """Gera gráfico do perfil de consumo energético e custo computacional."""
        plt.figure(figsize=(10, 4))
        plt.plot(steps, energy_values, color='tab:red', linewidth=2, marker='s', markersize=4)
        plt.title("NeuroCore - Perfil de Consumo Energético por Ciclo", fontsize=12, fontweight='bold')
        plt.xlabel("Passos de Simulação", fontsize=10)
        plt.ylabel("Energia Estimada (Joules)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[Plotter] Gráfico de energia salvo em: {filepath}")