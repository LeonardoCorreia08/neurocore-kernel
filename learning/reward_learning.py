'''Modulo: learning/reward_learning.py'''
import torch
import torch.nn as nn

class RSTDPManager(nn.Module):
    def __init__(self, num_posts: int, num_pres: int, lr: float = 0.01, tau_trace: float = 20.0, tau_eligibility: float = 50.0):
        super().__init__()
        self.num_posts = num_posts
        self.num_pres = num_pres
        self.lr = lr
        
        # Constantes pré-calculadas
        self.decay_trace = torch.exp(torch.tensor(-1.0 / tau_trace))
        self.decay_eligibility = torch.exp(torch.tensor(-1.0 / tau_eligibility))
        
        self.A_plus = 1.0
        self.A_minus = -1.0
        
        # Tracos em tensores 1D alinhados ao hardware
        self.register_buffer('trace_pre', torch.zeros(num_pres))
        self.register_buffer('trace_post', torch.zeros(num_posts))
        
        # MELHORIA 12 / ITEM 22: Rastro de elegibilidade 1D focado estritamente nas sinapses ativas
        self.register_buffer('eligibility_trace', None)

    def reset_state(self):
        self.trace_pre.zero_()
        self.trace_post.zero_()
        if self.eligibility_trace is not None:
            self.eligibility_trace.zero_()

    def _init_eligibility_if_needed(self, num_connections: int, device: torch.device):
        """Garante que a elegibilidade acompanha o número exato de sinapses reais."""
        if self.eligibility_trace is None or self.eligibility_trace.shape[0] != num_connections:
            self.eligibility_trace = torch.zeros(num_connections, device=device)

    def step_eligibility(self, spk_pre: torch.Tensor, spk_post: torch.Tensor, sparse_indices: torch.Tensor):
        """Atualiza os tracos e a elegibilidade apenas para as conexões esparsas existentes."""
        spk_pre = spk_pre.view(-1)
        spk_post = spk_post.view(-1)

        # 1. Atualizar tracos temporais
        self.trace_pre = self.trace_pre * self.decay_trace + spk_pre
        self.trace_post = self.trace_post * self.decay_trace + spk_post
        
        # 2. Extrai apenas os cruzamentos válidos da rede
        idx_post = sparse_indices[0, :]
        idx_pre  = sparse_indices[1, :]
        
        self._init_eligibility_if_needed(sparse_indices.shape[1], spk_pre.device)
        
        # 3. Calcular mudanca instantanea por STDP (sem usar torch.outer)
        ltp = self.A_plus * (self.trace_pre[idx_pre] * spk_post[idx_post])
        ltd = self.A_minus * (self.trace_post[idx_post] * spk_pre[idx_pre])
        
        # 4. Atualizar rastro de elegibilidade com decaimento
        self.eligibility_trace = self.eligibility_trace * self.decay_eligibility + (ltp + ltd)

    def apply_reward_modulation(self, reward: float, sparse_indices: torch.Tensor) -> torch.Tensor:
        """
        [ITEM 22 OTIMIZADO]: Modulação Dopaminérgica puramente 1D.
        Retorna diretamente o vetor de deltas sem criar tensores esparsos intermediários.
        """
        if self.eligibility_trace is None or reward == 0.0:
            delta_values = torch.zeros(sparse_indices.shape[1], device=sparse_indices.device)
        else:
            # Delta W = lr * Reward * Eligibility_Trace
            delta_values = self.eligibility_trace * reward * self.lr
            
            # Zera o rastro apos a consolidacao do evento de recompensa
            self.eligibility_trace.zero_()
            
        # Retorna o vetor 1D direto para o update_sparse_weights da SparseSynapseNetwork
        return delta_values