from __future__ import annotations
import numpy as np


class LinearTDCritic:
    """Semi-gradient TD(0) com features assinadas e atualização normalizada."""

    def __init__(self, gamma: float = 0.95, learning_rate: float = 0.05):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma precisa estar em [0, 1]")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate precisa estar em (0, 1]")
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.weights: np.ndarray | None = None

    @staticmethod
    def _features(state) -> np.ndarray:
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("state deve conter features finitas e não vazias")
        return np.concatenate(([1.0], np.clip(values, -1.0, 1.0)))

    def _ensure_weights(self, feature_count: int) -> None:
        if self.weights is None:
            self.weights = np.zeros(feature_count, dtype=np.float64)
        elif self.weights.size != feature_count:
            raise ValueError(
                f"Dimensão incompatível do crítico: salvo={self.weights.size - 1}, "
                f"entrada={feature_count - 1}"
            )

    def value(self, state) -> float:
        features = self._features(state)
        self._ensure_weights(features.size)
        return float(np.dot(self.weights, features))

    def update(
        self,
        state,
        reward: float,
        next_state=None,
        done: bool = False,
        truncated: bool = False,
    ) -> float:
        """Atualiza o crítico e retorna δ = r + γV(s') − V(s).

        `done=True` zera o bootstrap. `truncated=True` continua bootstrapando,
        pois o limite externo não significa que a tarefa terminou naturalmente.
        """
        if not np.isfinite(reward):
            raise ValueError("reward precisa ser finito")
            
        features = self._features(state)
        self._ensure_weights(features.size)
        current_value = float(np.dot(self.weights, features))

        if done or next_state is None:
            next_value = 0.0
        else:
            next_features = self._features(next_state)
            self._ensure_weights(next_features.size)
            next_value = float(np.dot(self.weights, next_features))

        bootstrap = 0.0 if done else self.gamma * next_value
        td_error = float(reward) + bootstrap - current_value
        normalizer = 1.0 + float(np.dot(features, features))
        self.weights += self.learning_rate * td_error * features / normalizer
        return td_error

    def state_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "weights": None if self.weights is None else self.weights.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        gamma = float(state["gamma"])
        learning_rate = float(state["learning_rate"])
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma inválido no snapshot")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate inválido no snapshot")
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        saved_weights = state.get("weights")
        if saved_weights is None:
            self.weights = None
            return
        weights = np.asarray(saved_weights, dtype=np.float64)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)):
            raise ValueError("Pesos do crítico inválidos no snapshot")
        self.weights = weights.copy()