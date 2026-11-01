'''Modulo: associative.py'''
import torch

class AssociativeMemoryNetwork:
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim
        # Matriz Hebbiana de correlacao associativa
        self.association_matrix = torch.zeros((vector_dim, vector_dim))

    def encode_association(self, pattern_a: torch.Tensor, pattern_b: torch.Tensor):
        """Associa dois padroes neurais (Hebbian learning: células que disparam juntas, se conectam)."""
        outer_prod = torch.outer(pattern_a, pattern_b)
        self.association_matrix += outer_prod
        # Normalizar para evitar estouro
        self.association_matrix = torch.clamp(self.association_matrix, min=-10.0, max=10.0)

    def recall(self, query_pattern: torch.Tensor) -> torch.Tensor:
        """Resgata o padrao associado com base em uma consulta parcial."""
        recalled = torch.matmul(self.association_matrix, query_pattern)
        return torch.tanh(recalled) # Ativacao delimitada