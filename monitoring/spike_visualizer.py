'''Modulo: spike_visualizer.py'''
import os
import matplotlib.pyplot as plt
import torch

class SpikeVisualizer:
    def __init__(self):
        self.spike_history = []

    def record_spikes(self, spike_tensor: torch.Tensor):
        """Registra os spikes de um passo de tempo para posterior visualizacao."""
        self.spike_history.append(spike_tensor.detach().cpu().clone())

    @staticmethod
    def plot_raster(spike_history=None, filename: str = "docs/spike_raster.png", title: str = "NeuroCore - Raster Plot de Atividade Neuronal"):
        """
        Suporta tanto chamada estática com lista quanto uso via instância.
        """
        visualizer = SpikeVisualizer()
        
        # Se foi passado uma lista ou iterável diretamente no argumento
        if spike_history is not None:
            if isinstance(spike_history, list):
                visualizer.spike_history = spike_history
            else:
                # Caso tenham chamado a partir de uma instância self
                visualizer.spike_history = getattr(spike_history, 'spike_history', [])
        
        # Se estiver usando via instância (self passado implicitamente como primeiro arg se não for staticmethod, 
        # mas como está como staticmethod/flexível, tratamos o histórico local)
        if not visualizer.spike_history:
            print("[Visualizer] Nenhum spike registrado para plotagem.")
            return

        spikes_tensor = torch.stack(visualizer.spike_history).detach().cpu().numpy()
        steps, neurons = spikes_tensor.shape

        plt.figure(figsize=(10, 4))
        
        for neuron_id in range(neurons):
            spike_times = [t for t in range(steps) if spikes_tensor[t, neuron_id] > 0]
            if len(spike_times) > 0:
                plt.scatter(spike_times, [neuron_id] * len(spike_times), marker='|', color='black', s=10)

        plt.title(title)
        plt.xlabel("Passos de Tempo (Steps)")
        plt.ylabel("ID do Neurônio")
        plt.xlim(0, steps)
        plt.ylim(-1, neurons)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Raster Plot salvo com sucesso em '{filename}'!")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_static(spike_history: list, filename: str = "docs/spike_raster.png"):
        """Método utilitário estático caso queira passar uma lista crua diretamente."""
        visualizer = SpikeVisualizer()
        visualizer.spike_history = spike_history
        visualizer.plot_raster(filename=filename)

    def reset(self):
        self.spike_history.clear()