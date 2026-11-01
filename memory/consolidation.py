'''Modulo: consolidation.py'''
import torch

class ConsolidationEngine:
    def __init__(self, pruning_threshold: float = 0.01):
        self.pruning_threshold = pruning_threshold

    def apply_active_forgetting(self, sparse_weights: torch.Tensor) -> torch.Tensor:
        """
        Remove sinapses cujos pesos estao abaixo do limiar de significancia,
        realizando o esquecimento ativo e liberando capacidade de rede.
        """
        indices = sparse_weights.indices()
        values = sparse_weights.values()
        
        # Filtrar apenas sinapses com relevancia acima do limiar
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