"""Testes unitários mínimos do crítico linear TD(0)."""

import numpy as np

from learning.td_critic import LinearTDCritic


def run_tests() -> None:
    critic = LinearTDCritic(gamma=0.9, learning_rate=0.1)
    terminal_error = critic.update([0.0], reward=1.0, done=True)
    assert terminal_error == 1.0
    assert critic.value([0.0]) > 0.0

    state_value = critic.value([0.0])
    next_value = critic.value([1.0])
    expected = 0.0 + 0.9 * next_value - state_value
    truncated_error = critic.update(
        [0.0], reward=0.0, next_state=[1.0], done=False, truncated=True
    )
    assert abs(truncated_error - expected) < 1e-12
    assert truncated_error > -0.01, "Truncamento externo deve bootstrapar V(next_state)."

    saved = critic.state_dict()
    restored = LinearTDCritic()
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved

    other = LinearTDCritic()
    other.value([0.0, 1.0])
    try:
        other.value([0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Dimensões incompatíveis do crítico deveriam falhar explicitamente.")

    print("Teste do crítico TD(0): PASSOU")


if __name__ == "__main__":
    run_tests()
