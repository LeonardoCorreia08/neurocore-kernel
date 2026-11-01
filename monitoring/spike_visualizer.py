'''Modulo: spike_visualizer.py'''
import torch
import matplotlib.pyplot as plt

class SpikeVisualizer:
    def __init__(self):
        self.spike_history = []

    def record_spikes(self, spike_tensor: torch.Tensor):
        """Registra os spikes de um passo de tempo para posterior visualizacao."""
        self.spike_history.append(spike_tensor.detach().cpu().clone())

    def plot_raster(self, title: str = "Neural Activity Raster Plot"):
        """Gera e exibe um grafico de raster plots com base no historico de spikes."""
        if not self.spike_history:
            print("[Visualizer] Nenhum spike registrado para plotagem.")
            return

        history_matrix = torch.stack(self.spike_history).numpy() # Forma: [tempo, neuronios]
        time_steps, num_neurons = history_matrix.shape

        plt.figure(figsize=(10, 4))
        for neuron_id in range(min(num_neurons, 200)):
            spike_times = torch.where(torch.tensor(history_matrix[:, neuron_id]) > 0)[0].numpy()
            if len(spike_times) > 0:
                plt.scatter(spike_times, [neuron_id] * len(spike_times), s=2, color='black')

        plt.title(title)
        plt.xlabel("Passos de Tempo (ms)")
        plt.ylabel("ID do Neurônio")
        plt.xlim(0, time_steps)
        plt.ylim(-1, min(num_neurons, 200))
        plt.tight_layout()
        plt.show()

    def reset(self):
        self.spike_history.clear()