'''Modulo: associative.py'''
import torch

class AssociativeMemoryNetwork:
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim
        # Matriz Hebbiana de correlacao associativa
        self.association_matrix = torch.zeros((vector_dim, vector_dim))

    def encode_association(self, pattern_a: torch.Tensor, pattern_b: torch.Tensor):
        """Associa dois padroes neurais (Hebbian learning)."""
        # CORREÇÃO: Para resgatar B a partir de A, a matriz deve ser B * A^T
        outer_prod = torch.outer(pattern_b, pattern_a)
        self.association_matrix += outer_prod
        # Normalizar para evitar estouro
        self.association_matrix = torch.clamp(self.association_matrix, min=-10.0, max=10.0)

    def recall(self, query_pattern: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Resgata o padrao associado com base em uma consulta parcial."""
        recalled = torch.matmul(self.association_matrix, query_pattern)
        # Transforma o sinal bruto (ex: [0, 2, 0, 2]) de volta em spikes binários [0, 1, 0, 1]
        return (recalled > threshold).float()