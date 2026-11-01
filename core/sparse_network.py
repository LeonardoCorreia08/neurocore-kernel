'''Modulo: sparse_network.py'''
import torch

class SparseSynapseNetwork:
    def __init__(self, num_posts: int, num_pres: int, density: float = 0.15, weight_scale: float = 0.1):
        self.num_posts = num_posts
        self.num_pres = num_pres
        self.density = density
        
        # Gerar indices de conexao esparsa aleatorios (formato COO do PyTorch)
        total_connections = int(num_posts * num_pres * density)
        
        # Criar indices aleatorios para linhas (posts) e colunas (pres)
        row_indices = torch.randint(0, num_posts, (total_connections,))
        col_indices = torch.randint(0, num_pres, (total_connections,))
        indices = torch.stack([row_indices, col_indices], dim=0)
        
        # Inicializar os pesos sinapticos apenas para as conexoes ativas existentes
        values = torch.randn(total_connections) * weight_scale
        
        # Criar a matriz esparsa em memoria (Sparse Tensor)
        self.sparse_weights = torch.sparse_coo_tensor(
            indices, values, size=(num_posts, num_pres)
        ).coalesce()

    def get_dense_weights(self) -> torch.Tensor:
        """Converte temporariamente para denso para operacoes de matriz se necessario."""
        return self.sparse_weights.to_dense()

    def forward_propagation(self, spk_pre: torch.Tensor) -> torch.Tensor:
        """
        Realiza o produto matriz-vetor esparso: W * spk_pre
        Calcula eficientemente a corrente gerada apenas pelas sinapses ativas.
        """
        # Multiplicacao matriz esparsa x vetor de spikes
        return torch.sparse.mm(self.sparse_weights, spk_pre.unsqueeze(1)).squeeze(1)

    def update_sparse_weights(self, delta_weights: torch.Tensor):
        """
        Atualiza os valores dos pesos esparsos de forma segura, 
        garantindo compatibilidade de tamanho com as conexoes ativas.
        """
        # Se o delta veio denso (tamanho [num_posts, num_pres]), extraimos apenas 
        # os valores correspondentes aos indices esparsos existentes.
        if not delta_weights.is_sparse:
            indices = self.sparse_weights.indices()
            # Pegar os deltas nas posicoes exatas [row, col] da matriz esparsa
            delta_values = delta_weights[indices[0], indices[1]]
        else:
            delta_values = delta_weights.values()

        current_values = self.sparse_weights.values() + delta_values
        current_values = torch.clamp(current_values, min=-1.0, max=1.0)
        
        self.sparse_weights = torch.sparse_coo_tensor(
            self.sparse_weights.indices(), current_values, size=self.sparse_weights.size()
        ).coalesce()