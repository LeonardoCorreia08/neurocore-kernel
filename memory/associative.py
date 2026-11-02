'''Modulo: associative.py'''
import torch
from core.config_loader import get_config_value


class AssociativeMemoryNetwork:
    def __init__(self, vector_dim: int | None = None, config: dict | None = None):
        configured_dim = get_config_value(config, "memory.associative.vector_dim", None)
        self.vector_dim = vector_dim if vector_dim is not None else configured_dim
        if self.vector_dim is None:
            raise TypeError("vector_dim é obrigatório quando memory.associative.vector_dim não está configurado")
        self.recall_threshold = get_config_value(config, "memory.associative.recall_threshold", 0.5)
        self.weight_min = get_config_value(config, "memory.associative.weight_min", -10.0)
        self.weight_max = get_config_value(config, "memory.associative.weight_max", 10.0)
        # Matriz Hebbiana de correlacao associativa
        self.association_matrix = torch.zeros((self.vector_dim, self.vector_dim))

    def encode_association(self, pattern_a: torch.Tensor, pattern_b: torch.Tensor):
        """Associa dois padroes neurais (Hebbian learning)."""
        # Para resgatar B a partir de A, a matriz deve ser B * A^T
        outer_prod = torch.outer(pattern_b, pattern_a)
        self.association_matrix += outer_prod
        self.association_matrix = torch.clamp(self.association_matrix, min=self.weight_min, max=self.weight_max)

    def recall(self, query_pattern: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
        """Resgata o padrao associado com base em uma consulta parcial."""
        recalled = torch.matmul(self.association_matrix, query_pattern)
        threshold = self.recall_threshold if threshold is None else threshold
        return (recalled > threshold).float()
