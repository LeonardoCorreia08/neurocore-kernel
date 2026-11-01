"""Testes rápidos da representação direcional e da geometria do VirtualWorld2D."""

import numpy as np

from environment.world import VirtualWorld2D


def run_tests() -> None:
    world = VirtualWorld2D(width=120, height=60, sensor_mode="directional")
    for _ in range(100):
        state = world.reset()
        assert state.shape == (6,)
        assert np.linalg.norm(world.agent_pos - world.goal_pos) > 20.0
        assert world._is_obstacle_safe(world.agent_pos)
        assert world._is_obstacle_safe(world.goal_pos, clearance=world.goal_radius)

    world.agent_pos = np.array([1.0, 30.0])
    world.goal_pos = np.array([110.0, 30.0])
    state = world.get_sensors()
    assert state[0] > 0.0, "A direção do objetivo deve indicar +x."
    assert state[4] > 0.0, "Perto da parede esquerda, a repulsão deve apontar para +x."

    world.agent_pos = np.array([25.0, 40.0])
    world.goal_pos = np.array([100.0, 40.0])
    state = world.get_sensors()
    assert state[2] < 0.0, "Com obstáculo à direita, a repulsão deve apontar para -x."

    world.agent_pos = np.array([119.0, 30.0])
    state = world.get_sensors()
    assert state[4] < 0.0, "Perto da parede direita, a repulsão deve apontar para -x."

    world.agent_pos = np.array([60.0, 59.0])
    world.goal_pos = np.array([60.0, 10.0])
    world.step(0)
    assert world.agent_pos[1] == 60.0, "O eixo y deve respeitar height em mapa não quadrado."

    compact = VirtualWorld2D(sensor_mode="compact")
    assert compact.reset().shape == (4,), "O modo compacto legado deve continuar disponível."

    try:
        world.step(4)
    except ValueError:
        pass
    else:
        raise AssertionError("Ações fora de [0, 3] deveriam ser rejeitadas.")

    print("Teste de sensores direcionais e geometria: PASSOU")


if __name__ == "__main__":
    run_tests()
