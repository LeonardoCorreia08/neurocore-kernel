'''Modulo: synapse.py'''
import torch

class SparseSynapseManager:
    def __init__(self, num_posts: int, num_pres: int, density: float = 0.15, weight_scale: float = 0.2):
        self.num_posts = num_posts
        self.num_pres = num_pres
        self.density = density
        
        total_connections = int(num_posts * num_pres * density)
        row_indices = torch.randint(0, num_posts, (total_connections,))
        col_indices = torch.randint(0, num_pres, (total_connections,))
        indices = torch.stack([row_indices, col_indices], dim=0)
        values = torch.randn(total_connections) * weight_scale
        
        self.sparse_weights = torch.sparse_coo_tensor(
            indices, values, size=(num_posts, num_pres)
        ).coalesce()

    def forward_propagation(self, spk_pre: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.sparse_weights, spk_pre.unsqueeze(1)).squeeze(1)

    def update_sparse_weights(self, delta_weights: torch.Tensor):
        if not delta_weights.is_sparse:
            indices = self.sparse_weights.indices()
            delta_values = delta_weights[indices[0], indices[1]]
        else:
            delta_values = delta_weights.values()

        current_values = self.sparse_weights.values() + delta_values
        current_values = torch.clamp(current_values, min=-1.0, max=1.0)
        
        self.sparse_weights = torch.sparse_coo_tensor(
            self.sparse_weights.indices(), current_values, size=self.sparse_weights.size()
        ).coalesce()

    def save_checkpoint(self, filepath: str):
        torch.save({
            'indices': self.sparse_weights.indices(),
            'values': self.sparse_weights.values(),
            'size': self.sparse_weights.size()
        }, filepath)
        print(f"Checkpoint sinaptico salvo com sucesso em '{filepath}'!")