'''Modulo: test_14_generate_plots.py'''
import torch
import random
from monitoring.experiment_plotter import ExperimentPlotter

def generate_experiment_charts():
    print("Iniciando geração de gráficos dos experimentos do NeuroCore...")
    plotter = ExperimentPlotter(output_dir="experiments/plots")
    
    # 1. Simular e plotar Curva de Aprendizado (ex: 50 episódios)
    print("Gerando curva de aprendizado...")
    simulated_rewards = []
    accumulated = 0
    for ep in range(50):
        if ep > 5 and random.random() > 0.4:
            accumulated += random.randint(1, 4)
        simulated_rewards.append(float(accumulated))
    plotter.plot_learning_curve(simulated_rewards, filename="neurocore_learning_curve.png")
    
    # 2. Simular e plotar Raster Plot de Spikes (ex: 200 passos de tempo, 100 neurônios)
    print("Gerando raster plot de spikes...")
    simulated_spike_history = []
    for t in range(200):
        # Simula disparos esparsos na rede
        spikes = (torch.rand(100) > 0.85).float()
        simulated_spike_history.append(spikes)
    plotter.plot_raster(simulated_spike_history, filename="neurocore_raster_plot.png", max_neurons=100)
    
    # 3. Simular e plotar Perfil Energético
    print("Gerando perfil energético...")
    steps = list(range(10, 101, 10))
    energy_vals = [e * 1e-7 + random.uniform(1e-8, 5e-8) for e in range(len(steps))]
    plotter.plot_energy_and_sops(steps, energy_vals, filename="neurocore_energy_profile.png")
    
    print("Todos os gráficos foram gerados e salvos com sucesso na pasta 'experiments/plots/'!")

if __name__ == "__main__":
    generate_experiment_charts()