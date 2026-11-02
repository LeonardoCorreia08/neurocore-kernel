'''Modulo: core/simulation.py'''
import torch

from core.config_loader import get_config_value
from core.spike_engine import SpikeEngine

class TemporalSimulator:
    def __init__(self, engine: SpikeEngine, time_steps: int = 20, config: dict | None = None):
        """
        Gerencia a execução temporal da SNN. 
        Mapeia a interação entre o Mundo (Estático) e o Cérebro (Temporal).
        
        Args:
            engine: O motor central da rede (SpikeEngine).
            time_steps: Quantos milissegundos (ticks) o cérebro tem para processar 
                        cada estado do ambiente antes de tomar uma decisão.
        """
        self.engine = engine
        self.time_steps = get_config_value(config, "simulation.time_steps", time_steps)
        if self.time_steps <= 0:
            raise ValueError("simulation.time_steps precisa ser positivo.")

    def predict_action(self, sensory_array: list[float] | torch.Tensor) -> int:
        """
        Injeta os sensores na rede, simula a passagem do tempo e decide a ação 
        baseada na "Rate Coding" (qual neurônio motor disparou mais).
        """
        # Converte a entrada do ambiente em Tensor de corrente
        if not isinstance(sensory_array, torch.Tensor):
            sensory_current = torch.tensor(sensory_array, dtype=torch.float32)
        else:
            sensory_current = sensory_array.clone().detach().float()
        sensory_current = sensory_current.to(self.engine.device)

        # Descobre quantos neurônios motores existem na última camada
        num_motor_neurons = self.engine.neurons[-1].num_neurons
        
        # Inicia um acumulador de "votos" para os neurônios motores
        motor_spike_counts = torch.zeros(
            num_motor_neurons,
            device=self.engine.device,
            dtype=sensory_current.dtype,
        )

        # ---------------------------------------------------------
        # LOOP TEMPORAL DA SNN
        # ---------------------------------------------------------
        for tick in range(self.time_steps):
            # O motor processa 1 tick completo (inclui homeostase e aprendizagem contínua)
            all_layer_spikes = self.engine(sensory_current)
            
            # Captura a atividade da última camada (Motora)
            motor_spikes = all_layer_spikes[-1]
            motor_spike_counts += motor_spikes

        # ---------------------------------------------------------
        # DECISÃO DE AÇÃO
        # ---------------------------------------------------------
        # Se nenhum neurônio disparou, a ação 0 é escolhida por padrão. 
        # (A Homeostase garantirá que eles fiquem mais sensíveis no futuro caso isto aconteça muito)
        action_id = torch.argmax(motor_spike_counts).item()
        
        return action_id

    def process_feedback(self, reward: float):
        """
        Recebe a avaliação do ambiente e injeta no Cérebro (Dopamina).
        """
        self.engine.apply_global_reward(reward)

    def reset_for_new_episode(self):
        """
        Limpa todos os estados elétricos e traços temporais para iniciar 
        uma nova vida/episódio no ambiente virtual.
        """
        self.engine.reset_all_states()
