'''Modulo: stdp.py'''
import torch

class STDPManager:
    def __init__(self, num_posts: int, num_pres: int, lr: float = 0.005, tau_trace: float = 20.0):
        self.lr = lr
        self.tau_trace = tau_trace
        
        # Tracos de atividade para neuronios pre e pos-sinapticos
        self.trace_pre = torch.zeros(num_pres)
        self.trace_post = torch.zeros(num_posts)
        
        # Parametros de amplitude da STDP
        self.A_plus = 1.0
        self.A_minus = -1.0

    def reset_traces(self):
        self.trace_pre.zero_()
        self.trace_post.zero_()

    def update_weights(self, weights: torch.Tensor, spk_pre: torch.Tensor, spk_post: torch.Tensor):
        """
        Atualiza a matriz de pesos sinapticos com base nos spikes atuais e traços temporais.
        weights: tensor de dimensao [num_posts, num_pres]
        spk_pre: tensor binario de spikes da camada anterior [num_pres]
        spk_post: tensor binario de spikes da camada atual [num_posts]
        """
        decay = torch.exp(torch.tensor(-1.0 / self.tau_trace))
        
        # Atualizar tracos temporais (decay + novos spikes)
        self.trace_pre = self.trace_pre * decay + spk_pre
        self.trace_post = self.trace_post * decay + spk_post
        
        # Regra de STDP baseada em tracos (trace-based STDP)
        # 1. Quando ocorre spike pos-sinaptico: recompensa conexoes pre recentes (A_plus)
        # 2. Quando ocorre spike pre-sinaptico: penaliza se o pos disparou antes (A_minus)
        
        delta_w = torch.outer(spk_post, self.trace_pre) * self.A_plus + \
                  torch.outer(self.trace_post, spk_pre) * self.A_minus
                  
        # Atualizar pesos aplicando a taxa de aprendizado e limitando os limites (ex: [0, 1])
        weights = weights + self.lr * delta_w
        weights = torch.clamp(weights, min=-1.0, max=1.0)
        
        return weights