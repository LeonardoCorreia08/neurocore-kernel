"""Gestão de memória episódica e snapshots persistentes do SpikeEngine."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch


class MemoryManager:
    FORMAT_VERSION = 2

    def __init__(self, engine, save_dir: str = "checkpoints/", max_buffer_size: int = 1000):
        self.engine = engine
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer_size = max(1, int(max_buffer_size))
        self.episodic_buffer: list[dict[str, Any]] = []
        self.experiences_seen = 0
        self.restored_runtime: dict[str, Any] = {}

    @staticmethod
    def _vector_to_list(value) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        return [float(x) for x in value]

    def store_experience(
        self,
        sensory_state,
        action: int,
        reward: float,
        next_state=None,
        done: bool = False,
        task_id: str | int | None = None,
        learning_reward: float | None = None,
        truncated: bool = False,
    ) -> None:
        """Store a complete transition using reservoir sampling when full."""
        experience = {
            "state": self._vector_to_list(sensory_state),
            "action": int(action),
            "reward": float(reward),
            "learning_reward": float(reward if learning_reward is None else learning_reward),
            "next_state": self._vector_to_list(next_state),
            "done": bool(done),
            "truncated": bool(truncated),
            "task_id": task_id,
            "sequence": self.experiences_seen,
        }
        self.experiences_seen += 1

        if len(self.episodic_buffer) < self.max_buffer_size:
            self.episodic_buffer.append(experience)
            return

        # Reservoir sampling gives each observed transition equal probability
        # of surviving a long stream, instead of deterministically deleting
        # only the oldest data as FIFO does.
        replacement = random.randrange(self.experiences_seen)
        if replacement < self.max_buffer_size:
            self.episodic_buffer[replacement] = experience

    def sample_experiences(
        self,
        batch_size: int,
        task_id: str | int | None = None,
        balanced: bool = True,
    ) -> list[dict[str, Any]]:
        """Sample without replacement; optionally balance action/task buckets."""
        if batch_size <= 0 or not self.episodic_buffer:
            return []

        candidates = [
            item for item in self.episodic_buffer
            if task_id is None or item.get("task_id") == task_id
        ]
        if not candidates:
            return []
        count = min(int(batch_size), len(candidates))
        if not balanced:
            return random.sample(candidates, count)

        buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for item in candidates:
            key = (str(item.get("task_id")), int(item["action"]))
            buckets.setdefault(key, []).append(item)
        keys = list(buckets)
        random.shuffle(keys)
        for bucket in buckets.values():
            random.shuffle(bucket)

        selected = []
        while len(selected) < count:
            active = [key for key in keys if buckets[key]]
            if not active:
                break
            for key in active:
                selected.append(buckets[key].pop())
                if len(selected) == count:
                    break
        return selected

    def clear_short_term_memory(self) -> None:
        """Explicitly clear episodes; normal consolidation no longer clears them."""
        self.episodic_buffer.clear()
        self.experiences_seen = 0

    def _path(self, memory_name: str) -> Path:
        candidate = Path(memory_name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("memory_name deve ser um nome relativo dentro de save_dir")
        return self.save_dir / candidate

    @staticmethod
    def _cpu_copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: MemoryManager._cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [MemoryManager._cpu_copy(item) for item in value]
        return value

    def _layer_sizes(self) -> list[int]:
        return [int(neuron.num_neurons) for neuron in self.engine.neurons]

    def _learning_config(self) -> dict[str, Any]:
        config = {
            "enable_stdp": bool(self.engine.enable_stdp),
            "enable_rstdp": bool(self.engine.enable_rstdp),
            "enable_homeostasis": bool(self.engine.enable_homeostasis),
            "neurons": [],
            "stdp": [],
            "rstdp": [],
            "homeostasis": [],
        }
        for neuron in self.engine.neurons:
            threshold = neuron.threshold
            config["neurons"].append({
                "threshold": self._cpu_copy(threshold),
                "decay": float(neuron.decay),
            })
        for manager in self.engine.stdp_managers:
            config["stdp"].append({
                "lr": float(manager.lr),
                "tau_trace": float(manager.tau_trace),
                "A_plus": float(manager.A_plus),
                "A_minus": float(manager.A_minus),
                "decay": self._cpu_copy(manager.decay),
            })
        for manager in self.engine.rstdp_managers:
            config["rstdp"].append({
                "lr": float(manager.lr),
                "A_plus": float(manager.A_plus),
                "A_minus": float(manager.A_minus),
                "decay_trace": self._cpu_copy(manager.decay_trace),
                "decay_eligibility": self._cpu_copy(manager.decay_eligibility),
            })
        for manager in self.engine.homeostasis_managers:
            config["homeostasis"].append({
                "target_rate": float(manager.target_rate),
                "adjustment_rate": float(manager.adjustment_rate),
                "decay": self._cpu_copy(manager.decay),
            })
        return config

    def _build_snapshot(self, runtime_metadata: dict[str, Any] | None) -> dict[str, Any]:
        synapses = []
        for layer in self.engine.synapses:
            weights = layer.sparse_weights.coalesce()
            synapses.append({
                "indices": weights.indices().detach().cpu().clone(),
                "values": weights.values().detach().cpu().clone(),
                "size": tuple(int(dim) for dim in weights.size()),
                "density": float(getattr(layer, "density", 0.0)),
            })

        neurons = []
        for neuron in self.engine.neurons:
            neurons.append({
                "mem": self._cpu_copy(neuron.mem),
            })

        return {
            "format_version": self.FORMAT_VERSION,
            "torch_version": str(torch.__version__),
            "architecture": self._layer_sizes(),
            "synapses": synapses,
            "engine_state": self._cpu_copy(self.engine.state_dict()),
            "neuron_state": neurons,
            "learning_config": self._learning_config(),
            "episodes": self._cpu_copy(self.episodic_buffer),
            "experiences_seen": int(self.experiences_seen),
            "max_buffer_size": int(self.max_buffer_size),
            "runtime_metadata": runtime_metadata or {},
        }

    def consolidate_memory(
        self,
        memory_name: str = "latest_brain_engram.pt",
        runtime_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Atomically save topology, weights, plastic state, config and episodes."""
        filepath = self._path(memory_name)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self._build_snapshot(runtime_metadata)
        fd, temp_name = tempfile.mkstemp(prefix=f".{filepath.name}.", suffix=".tmp", dir=filepath.parent)
        os.close(fd)
        try:
            torch.save(snapshot, temp_name)
            os.replace(temp_name, filepath)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return str(filepath)

    def _restore_learning_config(self, config: dict[str, Any]) -> None:
        self.engine.enable_stdp = bool(config["enable_stdp"])
        self.engine.enable_rstdp = bool(config["enable_rstdp"])
        self.engine.enable_homeostasis = bool(config["enable_homeostasis"])
        for neuron, saved in zip(self.engine.neurons, config["neurons"]):
            threshold = saved["threshold"]
            if isinstance(threshold, torch.Tensor):
                threshold = threshold.to(self.engine.device)
            neuron.threshold = threshold
            neuron.decay = float(saved["decay"])
        for manager, saved in zip(self.engine.stdp_managers, config["stdp"]):
            manager.lr = float(saved["lr"])
            manager.tau_trace = float(saved["tau_trace"])
            manager.A_plus = float(saved["A_plus"])
            manager.A_minus = float(saved["A_minus"])
            manager.decay = saved["decay"].to(self.engine.device)
        for manager, saved in zip(self.engine.rstdp_managers, config["rstdp"]):
            manager.lr = float(saved["lr"])
            manager.A_plus = float(saved["A_plus"])
            manager.A_minus = float(saved["A_minus"])
            manager.decay_trace = saved["decay_trace"].to(self.engine.device)
            manager.decay_eligibility = saved["decay_eligibility"].to(self.engine.device)
        for manager, saved in zip(self.engine.homeostasis_managers, config["homeostasis"]):
            manager.target_rate = float(saved["target_rate"])
            manager.adjustment_rate = float(saved["adjustment_rate"])
            manager.decay = saved["decay"].to(self.engine.device)

    def _retrieve_legacy(self, snapshot: dict[str, Any]) -> bool:
        """Load old weight-only snapshots produced by format version 1."""
        restored = []
        for index, layer in enumerate(self.engine.synapses):
            key = f"layer_{index}"
            if key not in snapshot:
                raise ValueError(f"Snapshot legado sem {key}")
            item = snapshot[key]
            expected_size = tuple(layer.sparse_weights.shape)
            if tuple(item["size"]) != expected_size:
                raise ValueError(f"Dimensão incompatível em {key}")
            restored.append(torch.sparse_coo_tensor(
                item["indices"].to(self.engine.device),
                item["values"].to(self.engine.device),
                size=expected_size,
                device=self.engine.device,
                check_invariants=True,
            ).coalesce())
        for layer, weights in zip(self.engine.synapses, restored):
            layer.sparse_weights = weights
        self.episodic_buffer = []
        self.experiences_seen = 0
        self.restored_runtime = {}
        return True

    def retrieve_memory(self, memory_name: str = "latest_brain_engram.pt") -> bool:
        """Validate then restore a snapshot. Safe for tensor/primitive snapshots."""
        filepath = self._path(memory_name)
        if not filepath.exists():
            return False
        snapshot = torch.load(filepath, map_location="cpu", weights_only=True)
        if not isinstance(snapshot, dict):
            raise ValueError("Formato de snapshot inválido")
        if "format_version" not in snapshot:
            return self._retrieve_legacy(snapshot)
        if snapshot["format_version"] != self.FORMAT_VERSION:
            raise ValueError(f"Versão de snapshot não suportada: {snapshot['format_version']}")
        if snapshot["architecture"] != self._layer_sizes():
            raise ValueError(
                f"Arquitetura incompatível: arquivo={snapshot['architecture']} "
                f"motor={self._layer_sizes()}"
            )
        if len(snapshot["synapses"]) != len(self.engine.synapses):
            raise ValueError("Número de camadas sinápticas incompatível")

        restored_weights = []
        for index, (saved, layer) in enumerate(zip(snapshot["synapses"], self.engine.synapses)):
            expected_size = tuple(int(d) for d in layer.sparse_weights.shape)
            if tuple(saved["size"]) != expected_size:
                raise ValueError(f"Dimensão incompatível na camada {index}")
            indices = saved["indices"]
            values = saved["values"]
            if indices.ndim != 2 or indices.shape[0] != 2 or values.ndim != 1:
                raise ValueError(f"Tensor esparso malformado na camada {index}")
            if indices.shape[1] != values.numel():
                raise ValueError(f"Índices e valores desalinhados na camada {index}")
            if values.numel() and not torch.isfinite(values).all():
                raise ValueError(f"Pesos não finitos na camada {index}")
            for axis, upper in enumerate(expected_size):
                axis_indices = indices[axis, :]
                if axis_indices.numel() and (axis_indices.min() < 0 or axis_indices.max() >= upper):
                    raise ValueError(f"Índice fora dos limites na camada {index}")
            restored_weights.append(torch.sparse_coo_tensor(
                indices.to(self.engine.device), values.to(self.engine.device),
                size=expected_size, device=self.engine.device,
                check_invariants=True,
            ).coalesce())

        config = snapshot["learning_config"]
        for key in ("neurons", "stdp", "rstdp", "homeostasis"):
            expected = {
                "neurons": len(self.engine.neurons),
                "stdp": len(self.engine.stdp_managers),
                "rstdp": len(self.engine.rstdp_managers),
                "homeostasis": len(self.engine.homeostasis_managers),
            }[key]
            if len(config[key]) != expected:
                raise ValueError(f"Configuração incompatível: {key}")

        # Materialize lazily-created eligibility buffers before loading state_dict.
        saved_engine_state = snapshot["engine_state"]
        for index, manager in enumerate(self.engine.rstdp_managers):
            name = f"rstdp_managers.{index}.eligibility_trace"
            if name in saved_engine_state and manager.eligibility_trace is None:
                manager.eligibility_trace = torch.zeros_like(
                    saved_engine_state[name], device=self.engine.device
                )

        current_state = self.engine.state_dict()
        if set(current_state) != set(saved_engine_state):
            missing = sorted(set(current_state) - set(saved_engine_state))
            unexpected = sorted(set(saved_engine_state) - set(current_state))
            raise ValueError(f"Estado interno incompatível; ausentes={missing}, extras={unexpected}")
        for key, current in current_state.items():
            saved = saved_engine_state[key]
            if current.shape != saved.shape:
                raise ValueError(f"Dimensão incompatível no estado interno {key}")

        # All structural checks passed; now apply the snapshot.
        self._restore_learning_config(config)
        for layer, weights in zip(self.engine.synapses, restored_weights):
            layer.sparse_weights = weights
        self.engine.load_state_dict(saved_engine_state, strict=True)
        for neuron, saved in zip(self.engine.neurons, snapshot["neuron_state"]):
            mem = saved["mem"]
            neuron.mem = None if mem is None else mem.to(self.engine.device)

        self.episodic_buffer = list(snapshot["episodes"])
        self.experiences_seen = int(snapshot["experiences_seen"])
        self.max_buffer_size = int(snapshot["max_buffer_size"])
        self.restored_runtime = dict(snapshot.get("runtime_metadata", {}))
        return True
