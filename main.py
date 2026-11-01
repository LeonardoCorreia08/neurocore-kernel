'''Modulo: main.py'''
import matplotlib.pyplot as plt
import numpy as np

from core.spike_engine import SpikeEngine
from core.simulation import TemporalSimulator
from memory.manager import MemoryManager
from perception.sensor_encoder import SensorEncoder
from learning.continual_learning import ContinualLearningAgent
from environment.world import VirtualWorld2D

def render_world(world, episode, step, ax):
    """Opção A: O 'Olho de Deus' - Desenha o ambiente 2D em tempo real."""
    ax.clear()
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_title(f"NeuroCore | Episódio: {episode} | Passo: {step}")
    
    # Desenha Obstáculos (Vermelho)
    for obs in world.obstacles:
        circle = plt.Circle((obs[0], obs[1]), world.obstacle_radius, color='red', alpha=0.6)
        ax.add_patch(circle)
        
    # Desenha Alvo (Verde)
    goal = plt.Circle((world.goal_pos[0], world.goal_pos[1]), world.goal_radius, color='green', alpha=0.6)
    ax.add_patch(goal)
    
    # Desenha Agente (Azul)
    agent = plt.Circle((world.agent_pos[0], world.agent_pos[1]), 2.0, color='blue', alpha=0.9)
    ax.add_patch(agent)
    
    plt.draw()
    plt.pause(0.05) # Pausa de 50ms para criar a animação fluida

def run_training_and_visualization():
    print("="*55)
    print("🧠 INICIANDO A MÁQUINA DE APRENDER (1000 Episódios)")
    print("="*55)

    # 1. Fluxo Único Integrado
    world = VirtualWorld2D(width=100, height=100)
    brain_engine = SpikeEngine(layer_sizes=[4, 32, 4], density=0.25)
    simulator = TemporalSimulator(engine=brain_engine, time_steps=20)
    memory = MemoryManager(engine=brain_engine, save_dir="checkpoints/")
    encoder = SensorEncoder(max_rate=100.0)
    agent = ContinualLearningAgent(simulator, memory, encoder)

    # 2. Configuração do Treino (Opção B)
    NUM_EPISODES = 1000
    MAX_STEPS = 50
    history_rewards = []
    
    # Prepara o modo interativo do Matplotlib
    plt.ion() 
    fig, ax = plt.subplots(figsize=(6, 6))

    print("-> Treinando em alta velocidade (Renderização ativada nos últimos 5 episódios)...\n")
    
    for ep in range(1, NUM_EPISODES + 1):
        state = world.reset()
        ep_reward = 0.0
        
        # Só ativa o "Olho de Deus" no fim para não atrasar o treino pesado
        render_live = (ep > NUM_EPISODES - 5)
        
        for step in range(1, MAX_STEPS + 1):
            action = agent.step(state)
            next_state, reward, done = world.step(action)
            agent.process_feedback(state, action, reward)
            
            ep_reward += reward
            state = next_state
            
            if render_live:
                render_world(world, ep, step, ax)
                
            if done:
                break
                
        history_rewards.append(ep_reward)
        
        if ep % 100 == 0:
            avg_rew = np.mean(history_rewards[-100:])
            print(f"Episódio {ep:04d}/{NUM_EPISODES} | Recompensa Média (últimos 100): {avg_rew:+.2f}")

    print("\n[+] Treino concluído! Agente vai dormir e consolidar memória...")
    agent.sleep()
    
    # 3. Gráfico Final da Curva de Aprendizagem
    plt.ioff()
    plt.close(fig)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history_rewards, alpha=0.3, color='blue', label='Recompensa Bruta')
    
    window = 50
    if len(history_rewards) >= window:
        moving_avg = np.convolve(history_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(history_rewards)), moving_avg, color='red', linewidth=2, label=f'Média Móvel ({window})')
        
    plt.title("Curva de Aprendizagem do NeuroCore (R-STDP / Homeostase)")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Total")
    plt.legend()
    plt.grid(True)
    plt.show() # Mantém o gráfico final aberto na tela

if __name__ == "__main__":
    run_training_and_visualization()