"""Leitura e helpers da configuração global do NeuroCore.

A configuração é carregada no ponto de composição (main.py) e injetada nos
módulos. Módulos de baixo nível não leem arquivos implicitamente, evitando
acoplamento ao diretório de execução e mantendo chamadas antigas compatíveis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "brain_config.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader que rejeita chaves repetidas no mesmo mapa YAML."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "ao construir um mapa YAML", node.start_mark,
                f"chave duplicada: {key!r}", key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def get_config_value(config: dict[str, Any] | None, dotted_path: str, default: Any) -> Any:
    """Retorna uma chave YAML aninhada; usa o default legado se faltar."""
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def get_config_section(config: dict[str, Any] | None, dotted_path: str) -> dict[str, Any]:
    """Obtém uma seção aninhada ou `{}` se ausente/não for mapa."""
    value = get_config_value(config, dotted_path, {})
    return value if isinstance(value, dict) else {}


def load_brain_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Carrega e valida o YAML; o caminho padrão independe do CWD."""
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML inválido em {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("A raiz de brain_config.yaml precisa ser um mapa YAML.")

    required_sections = (
        "runtime", "network", "neuron", "simulation", "learning",
        "environment", "perception", "brain", "memory", "monitoring",
    )
    missing = [key for key in required_sections if not isinstance(data.get(key), dict)]
    if missing:
        raise ValueError("Seções ausentes ou inválidas em brain_config.yaml: " + ", ".join(missing))

    layers = data["network"].get("layer_sizes")
    if not isinstance(layers, list) or len(layers) < 2 or any(type(n) is not int or n <= 0 for n in layers):
        raise ValueError("network.layer_sizes deve conter ao menos dois inteiros positivos.")
    density = float(data["network"].get("connectivity_density", 0))
    if not 0 < density <= 1:
        raise ValueError("network.connectivity_density precisa estar em (0, 1].")
    if int(data["runtime"].get("episodes", 0)) <= 0:
        raise ValueError("runtime.episodes precisa ser positivo.")
    if int(data["runtime"].get("max_steps_per_episode", 0)) <= 0:
        raise ValueError("runtime.max_steps_per_episode precisa ser positivo.")
    if int(data["simulation"].get("time_steps", 0)) <= 0:
        raise ValueError("simulation.time_steps precisa ser positivo.")
    if int(data["environment"].get("width", 0)) <= 0 or int(data["environment"].get("height", 0)) <= 0:
        raise ValueError("environment.width e height precisam ser positivos.")
    if not isinstance(data["learning"].get("stdp"), dict) or not isinstance(data["learning"].get("rstdp"), dict):
        raise ValueError("learning.stdp e learning.rstdp precisam ser mapas.")
    if not isinstance(data["learning"].get("homeostasis"), dict):
        raise ValueError("learning.homeostasis precisa ser um mapa.")

    # Registra o arquivo efetivo para metadados/logs, sem modificar o documento.
    data["_config_path"] = str(path)
    return data
