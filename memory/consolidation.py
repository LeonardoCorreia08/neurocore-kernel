'''Modulo: consolidation.py'''
import torch
from core.config_loader import get_config_value


class ConsolidationEngine:
    def __init__(self, pruning_threshold: float | None = None, config: dict | None = None):
        configured_threshold = get_config_value(config, "memory.consolidation_pruning_threshold", 0.01)
        self.pruning_threshold = pruning_threshold if pruning_threshold is not None else configured_threshold

    def apply_active_forgetting(self, sparse_weights: torch.Tensor) -> torch.Tensor:
        """
        Remove sinapses cujos pesos estao abaixo do limiar de significancia,
        realizando o esquecimento ativo e liberando capacidade de rede.
        """
        indices = sparse_weights.indices()
        values = sparse_weights.values()
        mask = torch.abs(values) >= self.pruning_threshold
        filtered_indices = indices[:, mask]
        filtered_values = values[mask]
        pruned_weights = torch.sparse_coo_tensor(
            filtered_indices, filtered_values, size=sparse_weights.size()
        ).coalesce()
        removed_count = values.numel() - filtered_values.numel()
        if removed_count > 0:
            print(f"[Consolidation] Esquecimento ativo aplicado: {removed_count} sinapses fracas podadas.")
        return pruned_weights
