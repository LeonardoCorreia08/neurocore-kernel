
"""Rede sináptica esparsa do NeuroCore."""

import torch
import torch.nn as nn


class SparseSynapseNetwork(nn.Module):
    def __init__(
        self,
        num_posts: int,
        num_pres: int,
        density: float = 0.15,
        weight_scale: float = 0.1,
    ):
        super().__init__()

        if num_posts <= 0 or num_pres <= 0:
            raise ValueError("O número de neurônios deve ser positivo.")

        if not 0.0 < density <= 1.0:
            raise ValueError("density precisa estar no intervalo (0, 1].")

        if weight_scale < 0:
            raise ValueError("weight_scale não pode ser negativo.")

        self.num_posts = num_posts
        self.num_pres = num_pres
        self.density = density

        total_possible = num_posts * num_pres
        target_connections = max(
            1,
            min(total_possible, int(total_possible * density)),
        )

        # Gera pares únicos diretamente, sem separar a ordenação
        # dos índices da associação com os pesos.
        if target_connections == total_possible:
            idx_post = torch.arange(num_posts).repeat_interleave(num_pres)
            idx_pre = torch.arange(num_pres).repeat(num_posts)
        else:
            linear_indices = torch.randperm(total_possible)[
                :target_connections
            ]

            idx_post = torch.div(
                linear_indices,
                num_pres,
                rounding_mode="floor",
            )
            idx_pre = linear_indices.remainder(num_pres)

        indices = torch.stack([idx_post, idx_pre]).long()

        values = torch.randn(target_connections) * weight_scale

        self.sparse_weights = torch.sparse_coo_tensor(
            indices,
            values,
            size=(num_posts, num_pres),
        ).coalesce()

    def get_dense_weights(self) -> torch.Tensor:
        """Retorna uma representação densa dos pesos."""
        return self.sparse_weights.to_dense()

    def forward_propagation(
        self,
        spk_pre: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula a corrente pós-sináptica W @ spikes."""
        if spk_pre.dim() == 1:
            spk_pre = spk_pre.unsqueeze(1)

        if spk_pre.dim() != 2:
            raise ValueError("spk_pre precisa ter uma ou duas dimensões.")

        if spk_pre.shape[0] != self.num_pres:
            raise ValueError(
                f"Entrada com {spk_pre.shape[0]} elementos; "
                f"esperados {self.num_pres}."
            )

        spk_pre = spk_pre.to(
            device=self.sparse_weights.device,
            dtype=self.sparse_weights.dtype,
        )

        current = torch.sparse.mm(self.sparse_weights, spk_pre)
        return current.squeeze(1)

    def update_sparse_weights(self, delta_values: torch.Tensor):
        """Atualiza os pesos mantendo a estrutura esparsa."""
        weights = self.sparse_weights.coalesce()
        indices = weights.indices()
        current_values = weights.values()

        delta_values = torch.as_tensor(
            delta_values,
            device=current_values.device,
            dtype=current_values.dtype,
        )

        if delta_values.is_sparse:
            delta_values = delta_values.coalesce().values()

        elif delta_values.dim() > 1:
            # Aceita uma matriz densa de deltas.
            if tuple(delta_values.shape) != (
                self.num_posts,
                self.num_pres,
            ):
                raise ValueError(
                    "A matriz de deltas possui dimensões incorretas."
                )

            delta_values = delta_values[
                indices[0],
                indices[1],
            ]

        if delta_values.numel() != current_values.numel():
            raise ValueError(
                f"Foram recebidos {delta_values.numel()} deltas, "
                f"mas existem {current_values.numel()} pesos."
            )

        delta_values = delta_values.reshape_as(current_values)

        updated_values = torch.clamp(
            current_values + delta_values,
            min=-1.0,
            max=1.0,
        )

        self.sparse_weights = torch.sparse_coo_tensor(
            indices,
            updated_values,
            size=weights.size(),
            device=updated_values.device,
        ).coalesce()

    def save_checkpoint(self, filepath: str):
        """Salva os índices, pesos e dimensões da rede."""
        weights = self.sparse_weights.coalesce()

        torch.save(
            {
                "indices": weights.indices().cpu(),
                "values": weights.values().cpu(),
                "size": tuple(weights.size()),
                "density": self.density,
            },
            filepath,
        )

        print(f"Checkpoint salvo em '{filepath}'.")

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        density: float = 0.15,
        map_location: str = "cpu",
    ):
        """Carrega uma rede esparsa de um checkpoint."""
        checkpoint = torch.load(
            filepath,
            map_location=map_location,
            weights_only=True,
        )

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.num_posts = checkpoint["size"][0]
        instance.num_pres = checkpoint["size"][1]
        instance.density = checkpoint.get("density", density)

        instance.sparse_weights = torch.sparse_coo_tensor(
            checkpoint["indices"],
            checkpoint["values"],
            size=checkpoint["size"],
        ).coalesce()

        return instance
