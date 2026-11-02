'''Modulo: learning/stdp.py'''
import torch
import torch.nn as nn

from core.config_loader import get_config_value


class STDPManager(nn.Module):
    def __init__(self, num_posts: int, num_pres: int, lr: float = 0.005, tau_trace: float = 20.0, config: dict | None = None):
        super().__init__()
        self.num_posts = num_posts
        self.num_pres = num_pres
        self.lr = get_config_value(config, "learning.stdp.lr", lr)
        self.tau_trace = get_config_value(config, "learning.stdp.tau_trace", tau_trace)

        # Parametros de amplitude da STDP
        self.A_plus = get_config_value(config, "learning.stdp.a_plus", 1.0)
        self.A_minus = get_config_value(config, "learning.stdp.a_minus", -1.0)

        # Buffer para acompanhar o dispositivo e dtype do módulo.
        self.register_buffer('decay', torch.exp(torch.tensor(-1.0 / self.tau_trace)))

        # Tracos de atividade em buffers para compatibilidade nativa com CPU/GPU (Batch-safe)
        self.register_buffer('trace_pre', torch.zeros(num_pres))
        self.register_buffer('trace_post', torch.zeros(num_posts))

    def reset_traces(self):
        self.trace_pre.zero_()
        self.trace_post.zero_()

    def compute_sparse_deltas(self, spk_pre: torch.Tensor, spk_post: torch.Tensor, sparse_indices: torch.Tensor) -> torch.Tensor:
        """
        Calcula as atualizações de pesos APENAS para as conexões ativas.
        Substitui o torch.outer() para evitar estouro de memória (O(N^2)).
        """
        # Achata os tensores para garantir compatibilidade estrutural
        spk_pre = spk_pre.view(-1)
        spk_post = spk_post.view(-1)

        # 1. Atualizar tracos temporais (decay + novos spikes)
        self.trace_pre = self.trace_pre * self.decay + spk_pre
        self.trace_post = self.trace_post * self.decay + spk_post

        # 2. Extrai os indices exatos onde existem conexoes reais na rede esparsa
        idx_post = sparse_indices[0, :]
        idx_pre  = sparse_indices[1, :]

        # 3. Regra de STDP baseada em tracos (trace-based STDP) restrita aos índices
        ltp = self.A_plus * (self.trace_pre[idx_pre] * spk_post[idx_post])
        ltd = self.A_minus * (self.trace_post[idx_post] * spk_pre[idx_pre])

        # 4. Calcula os valores finais do delta (ajuste)
        delta_values = (ltp + ltd) * self.lr

        # Retorna diretamente o vetor 1D ultraleve.
        return delta_values
