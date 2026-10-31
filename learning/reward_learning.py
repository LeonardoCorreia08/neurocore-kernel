'''Modulo: reward_learning.py'''
import torch

class RSTDPManager:
    def __init__(self, num_posts: int, num_pres: int, lr: float = 0.01, tau_trace: float = 20.0, tau_eligibility: float = 50.0):
        self.lr = lr
        self.tau_trace = tau_trace
        self.tau_eligibility = tau_eligibility
        
        # Tracos temporais para neuronios pre e pos-sinapticos
        self.trace_pre = torch.zeros(num_pres)
        self.trace_post = torch.zeros(num_posts)
        
        # Matriz de rastro de elegibilidade para cada sinapse [num_posts, num_pres]
        self.eligibility_trace = torch.zeros(num_posts, num_pres)
        
        self.A_plus = 1.0
        self.A_minus = -1.0

    def reset_state(self):
        self.trace_pre.zero_()
        self.trace_post.zero_()
        self.eligibility_trace.zero_()

    def step_eligibility(self, spk_pre: torch.Tensor, spk_post: torch.Tensor):
        """
        Atualiza os tracos de STDP e acumula o rastro de elegibilidade a cada passo de tempo,
        sem alterar diretamente os pesos ainda.
        """
        decay_trace = torch.exp(torch.tensor(-1.0 / self.tau_trace))
        decay_eligibility = torch.exp(torch.tensor(-1.0 / self.tau_eligibility))
        
        # Atualizar tracos temporais
        self.trace_pre = self.trace_pre * decay_trace + spk_pre
        self.trace_post = self.trace_post * decay_trace + spk_post
        
        # Calcular mudanca instantanea por STDP
        delta_stpd = torch.outer(spk_post, self.trace_pre) * self.A_plus + \
                     torch.outer(self.trace_post, spk_pre) * self.A_minus
                     
        # Atualizar rastro de elegibilidade com decaimento
        self.eligibility_trace = self.eligibility_trace * decay_eligibility + delta_stpd

    def apply_reward_modulation(self, weights: torch.Tensor, reward: float):
        """
        Modula a alteracao dos pesos usando o sinal global de recompensa (Dopamina simulada).
        Se reward > 0, consolida as sinapses elegíveis. Se reward <= 0, atenua.
        """
        if reward != 0.0:
            # W = W + lr * Reward * Eligibility_Trace
            weights = weights + self.lr * reward * self.eligibility_trace
            weights = torch.clamp(weights, min=-1.0, max=1.0)
            
            # Zerar o rastro apos a consolidacao do evento de recompensa
            self.eligibility_trace.zero_()
            
        return weights