'''Modulo: core/synapse.py'''
import torch
import torch.nn as nn
import warnings

# Filtro para ignorar o aviso de inibição de checagem esparsa do PyTorch
warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

class SparseSynapseManager(nn.Module):
    def __init__(self, num_posts: int, num_pres: int, density: float = 0.15, weight_scale: float = 0.2):
        super().__init__()
        self.num_posts = num_posts
        self.num_pres = num_pres
        self.density = density
        
        # 1. Definir o número de conexões baseado na densidade desejada
        total_possible = num_posts * num_pres
        num_connections = max(1, int(total_possible * density))
        
        # 2. Gerar índices (coordenadas) consistentes dentro dos limites corretos
        idx_post = torch.randint(0, num_posts, (num_connections,))
        idx_pre = torch.randint(0, num_pres, (num_connections,))
        indices = torch.stack([idx_post, idx_pre])
        
        # 3. Gerar valores iniciais (pesos)
        values = torch.randn(num_connections) * weight_scale
        
        # 4. Criar tensor esparso e GARANTIR consistência através do coalesce()
        # O coalesce() soma valores de índices duplicados acidentais e ordena a topologia
        raw_tensor = torch.sparse_coo_tensor(
            indices, values, size=(num_posts, num_pres)
        )
        self.sparse_weights = raw_tensor.coalesce()

    def forward_propagation(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """
        Calcula a injeção de corrente: Multiplicação matricial esparsa (W * S).
        """
        # Garante alinhamento de dimensões para o produto matricial
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(1)
            
        current_injection = torch.sparse.mm(self.sparse_weights, pre_spikes)
        
        return current_injection.squeeze(1)

    def update_sparse_weights(self, delta_weights: torch.Tensor, w_min: float = -1.0, w_max: float = 1.0):
        """
        Atualiza os pesos sem quebrar o alinhamento com as conexões (índices).
        """
        # 1. Extrai a topologia validada atual
        indices = self.sparse_weights.indices()
        current_values = self.sparse_weights.values()
        
        # 2. Aplica a atualização de aprendizado (ex: STDP)
        new_values = current_values + delta_weights
        
        # 3. Garante limites (Clamp)
        new_values = torch.clamp(new_values, min=w_min, max=w_max)
        
        # 4. Reconstrói o tensor de forma segura, acoplando índices e valores e forçando o coalesce
        self.sparse_weights = torch.sparse_coo_tensor(
            indices, new_values, size=self.sparse_weights.shape
        ).coalesce()

    def prune_weak_connections(self, threshold: float = 0.01):
        """
        Esquecimento Ativo: Elimina conexões fracas mantendo a consistência dos tensores.
        """
        indices = self.sparse_weights.indices()
        values = self.sparse_weights.values()
        
        # Encontra as conexões fortes que devem sobreviver
        mask = torch.abs(values) >= threshold
        
        # Filtra índices e valores de forma 100% sincronizada
        surviving_indices = indices[:, mask]
        surviving_values = values[mask]
        
        # Recria a rede limpa
        self.sparse_weights = torch.sparse_coo_tensor(
            surviving_indices, surviving_values, size=self.sparse_weights.shape
        ).coalesce()