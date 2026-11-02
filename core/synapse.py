'''Modulo: core/synapse.py'''
import torch
import torch.nn as nn
import warnings

from core.config_loader import get_config_value

# Filtro para ignorar o aviso de inibição de checagem esparsa do PyTorch
warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

class SparseSynapseManager(nn.Module):
    def __init__(self, num_posts: int, num_pres: int, density: float = 0.15, weight_scale: float = 0.2, config: dict | None = None):
        super().__init__()
        self.num_posts = num_posts
        self.num_pres = num_pres
        self.density = get_config_value(config, "network.legacy_synapse_manager.density", density)
        density = self.density
        weight_scale = get_config_value(config, "network.legacy_synapse_manager.weight_init_scale", weight_scale)
        self.weight_min = get_config_value(config, "network.legacy_synapse_manager.weight_min", -1.0)
        self.weight_max = get_config_value(config, "network.legacy_synapse_manager.weight_max", 1.0)
        self.weight_init_scale = weight_scale
        if self.weight_min > self.weight_max:
            raise ValueError("weight_min não pode ser maior que weight_max.")
        
        # 1. Definir o número de conexões baseado na densidade desejada
        total_possible = num_posts * num_pres
        num_connections = max(1, int(total_possible * density))
        
        # 2. Gerar índices (coordenadas) consistentes dentro dos limites corretos
        idx_post = torch.randint(0, num_posts, (num_connections,))
        idx_pre = torch.randint(0, num_pres, (num_connections,))
        indices = torch.stack([idx_post, idx_pre])
        
        # 3. Gerar valores iniciais (pesos)
        values = torch.randn(num_connections) * weight_scale
        values = values.clamp(min=self.weight_min, max=self.weight_max)
        
        # 4. Criar tensor esparso e GARANTIR consistência através do coalesce()
        # O coalesce() soma valores de índices duplicados acidentais e ordena a topologia
        raw_tensor = torch.sparse_coo_tensor(
            indices, values, size=(num_posts, num_pres)
        )
        self.register_buffer("sparse_weights", raw_tensor.coalesce())

    def forward_propagation(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """
        Calcula a injeção de corrente: Multiplicação matricial esparsa (W * S).
        """
        # Garante alinhamento de dimensões para o produto matricial
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(1)
            
        current_injection = torch.sparse.mm(self.sparse_weights, pre_spikes)
        
        return current_injection.squeeze(1)

    def update_sparse_weights(self, delta_weights: torch.Tensor, w_min: float | None = None, w_max: float | None = None, config: dict | None = None):
        """
        Atualiza os pesos sem quebrar o alinhamento com as conexões (índices).
        """
        # 1. Extrai a topologia validada atual
        indices = self.sparse_weights.indices()
        current_values = self.sparse_weights.values()
        
        # 2. Aplica a atualização de aprendizado (ex: STDP)
        delta_weights = torch.as_tensor(delta_weights, device=current_values.device, dtype=current_values.dtype)
        new_values = current_values + delta_weights
        
        # 3. Garante limites (Clamp)
        w_min = get_config_value(config, "network.legacy_synapse_manager.weight_min", self.weight_min) if w_min is None else w_min
        w_max = get_config_value(config, "network.legacy_synapse_manager.weight_max", self.weight_max) if w_max is None else w_max
        if w_min > w_max:
            raise ValueError("w_min não pode ser maior que w_max.")
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
