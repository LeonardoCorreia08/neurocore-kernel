'''Modulo: spike_visualizer.py'''
import matplotlib.pyplot as plt
import torch

class SpikeVisualizer:
    @staticmethod
    def plot_raster(spike_history: list, filename: str = "docs/spike_raster.png"):
        """
        spike_history: lista de tensores binarios de spikes ao longo do tempo [steps, num_neurons]
        """
        spikes_tensor = torch.stack(spike_history).detach().cpu().numpy() # [steps, neurons]
        
        plt.figure(figsize=(10, 4))
        steps, neurons = spikes_tensor.shape
        
        for neuron_id in range(neurons):
            # Encontrar os passos onde o neuronio disparou
            spike_times = [t for t in range(steps) if spikes_tensor[t, neuron_id] > 0]
            plt.scatter(spike_times, [neuron_id] * len(spike_times), marker='|', color='black', s=10)
            
        plt.title("NeuroCore - Raster Plot de Atividade Neuronal")
        plt.xlabel("Passos de Tempo (Steps)")
        plt.ylabel("ID do Neuronio")
        plt.xlim(0, steps)
        plt.ylim(-1, neurons)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Raster Plot salvo com sucesso em '{filename}'!")