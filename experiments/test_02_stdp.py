'''Modulo: test_02_stdp.py'''
import torch
import numpy as np
from core.neuron import LeakyIntegrateAndFire
from environment.world import VirtualWorld2D
from learning.stdp import STDPManager


# =====================================================================
# 1. TESTE UNITÁRIO: CAUSALIDADE TEMPORAL DO STDP (LTP vs LTD)
# =====================================================================
def run_stdp_synapse_test():
    print("Iniciando Teste Unitário de Plasticidade Sináptica (STDP)...")
    torch.manual_seed(42)

    stdp = STDPManager(num_posts=1, num_pres=1, lr=0.05)
    indices = torch.tensor([[0], [0]], dtype=torch.long)

    # --- Fase 1: Causalidade (Pré dispara em t=1, Pós dispara em t=2 -> LTP) ---
    stdp.reset_traces()
    # Passo 1: apenas Pré dispara (argumentos posicionais: pre, post, indices)
    dw_t1 = stdp.compute_sparse_deltas(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        indices
    )
    # Passo 2: apenas Pós dispara (logo após o Pré)
    dw_t2 = stdp.compute_sparse_deltas(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        indices
    )
    ltp_delta = dw_t2.item()
    print(f"Fase 1 (Pré -> Pós | LTP) | Delta W obtido: {ltp_delta:+.6f} (Esperado: > 0)")
    assert ltp_delta > 0.0, f"Falha no LTP: esperado Delta W positivo, mas obteve {ltp_delta}"

    # --- Fase 2: Anti-Causalidade (Pós dispara em t=1, Pré dispara em t=2 -> LTD) ---
    stdp.reset_traces()
    # Passo 1: apenas Pós dispara
    _ = stdp.compute_sparse_deltas(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        indices
    )
    # Passo 2: apenas Pré dispara (atrasado em relação ao Pós)
    dw_ltd = stdp.compute_sparse_deltas(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        indices
    )
    ltd_delta = dw_ltd.item()
    print(f"Fase 2 (Pós -> Pré | LTD) | Delta W obtido: {ltd_delta:+.6f} (Esperado: < 0)")
    assert ltd_delta < 0.0, f"Falha no LTD: esperado Delta W negativo, mas obteve {ltd_delta}"

    # --- Fase 3: Reset de Traços ---
    stdp.reset_traces()
    dw_zero = stdp.compute_sparse_deltas(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        indices
    )
    assert abs(dw_zero.item()) < 1e-7, "Falha no reset_traces: variação residual detectada."
    print("[PASS] Causalidade (LTP), Anti-causalidade (LTD) e Reset do STDP verificados!\n")


# =====================================================================
# 2. FUNÇÕES AUXILIARES DE DECISÃO E HOMEOSTASE SINÁPTICA
# =====================================================================
def select_motor_action(motor_spike_counts: torch.Tensor, mem_motor: torch.Tensor) -> int:
    """Decodifica a decisão motora via Rate Coding com desempate estocástico."""
    max_spikes = torch.max(motor_spike_counts).item()

    if max_spikes > 0:
        winners = torch.nonzero(motor_spike_counts == max_spikes, as_tuple=False).squeeze(-1)
        if winners.numel() == 1:
            return int(winners.item())
        chosen_idx = np.random.randint(0, winners.numel())
        return int(winners[chosen_idx].item())

    return int(np.random.randint(0, mem_motor.numel()))


def apply_synaptic_homeostasis(
    W: torch.Tensor, target_row_mean: float, w_min: float = -1.0, w_max: float = 1.0
) -> torch.Tensor:
    """
    Evita a explosão Hebbiana (Runaway Excitation) do STDP não-supervisionado:
    1. Aplica clipping rígido nos limites biológicos [w_min, w_max].
    2. Mantém a soma média de entrada de cada neurônio pós-sináptico ancorada em
       target_row_mean (competição sináptica: para uma sinapse subir, outra deve descer).
    """
    W_clamped = torch.clamp(W, min=w_min, max=w_max)
    current_row_mean = W_clamped.mean(dim=1, keepdim=True)
    W_normalized = W_clamped - current_row_mean + target_row_mean
    return torch.clamp(W_normalized, min=w_min, max=w_max)


# =====================================================================
# 3. EXPERIMENTO PRINCIPAL: SIMULAÇÃO NO AMBIENTE COM STDP ATIVO
# =====================================================================
def run_neurocore_02_stdp_experiment():
    print("Iniciando NeuroCore 0.1 - Experimento com STDP Ativo + Homeostase...")
    torch.manual_seed(42)
    np.random.seed(42)

    world = VirtualWorld2D()

    input_dim = 4
    hidden_dim = 16
    output_dim = 4
    snn_ticks_per_step = 5

    # Camadas LIF
    sensory_layer = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    motor_layer = LeakyIntegrateAndFire(num_neurons=output_dim)

    # Inicialização Fan-In calibrada com equalização de linha
    W1 = torch.randn(hidden_dim, input_dim) * (1.0 / np.sqrt(input_dim))
    W1 = apply_synaptic_homeostasis(W1, target_row_mean=0.12)

    W2 = torch.randn(output_dim, hidden_dim) * (1.0 / np.sqrt(hidden_dim))
    W2 = apply_synaptic_homeostasis(W2, target_row_mean=0.06)

    # Gerenciadores STDP:
    # W1 aprende representações sensoriais mais rápido (lr=0.005)
    # W2 tem plasticidade não-supervisionada bem suave (lr=0.001) para não viciar em paredes
    stdp_w1 = STDPManager(num_posts=hidden_dim, num_pres=input_dim, lr=0.005)
    stdp_w2 = STDPManager(num_posts=output_dim, num_pres=hidden_dim, lr=0.001)

    indices_w1 = torch.cartesian_prod(torch.arange(hidden_dim), torch.arange(input_dim)).T
    indices_w2 = torch.cartesian_prod(torch.arange(output_dim), torch.arange(hidden_dim)).T

    cortical_noise_std = 0.12
    num_episodes = 50
    total_goals_reached = 0

    for episode in range(num_episodes):
        sensors = world.reset()
        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()
        stdp_w1.reset_traces()
        stdp_w2.reset_traces()

        motor_fatigue = torch.zeros(output_dim)
        episode_reward = 0.0
        steps = 0

        ep_hidden_spikes = 0.0
        ep_motor_spikes = 0.0
        ep_abs_dw1 = 0.0
        ep_abs_dw2 = 0.0

        while steps < 200:
            x_sensory = torch.tensor(sensors, dtype=torch.float32)
            motor_spike_counts = torch.zeros(output_dim)

            for _ in range(snn_ticks_per_step):
                sensory_drive = x_sensory + (torch.randn(input_dim) * 0.05)
                spk_sensory, _ = sensory_layer(sensory_drive)

                h_input = torch.matmul(W1, spk_sensory) + (torch.randn(hidden_dim) * cortical_noise_std)
                spk_hidden, _ = hidden_layer(h_input)

                m_input = (
                    torch.matmul(W2, spk_hidden)
                    - motor_fatigue
                    + (torch.randn(output_dim) * cortical_noise_std)
                )
                spk_motor, mem_motor = motor_layer(m_input)

                # Fadiga motora e limite de hiperpolarização
                motor_fatigue = (motor_fatigue * 0.95) + (spk_motor * 0.15)
                hidden_layer.mem.clamp_(min=-0.5)
                motor_layer.mem.clamp_(min=-0.5)

                # Atualização STDP em tempo real a cada micro-tick (argumentos posicionais)
                delta_w1 = stdp_w1.compute_sparse_deltas(spk_sensory, spk_hidden, indices_w1).view_as(W1)
                delta_w2 = stdp_w2.compute_sparse_deltas(spk_hidden, spk_motor, indices_w2).view_as(W2)

                W1 = W1 + delta_w1
                W2 = W2 + delta_w2

                # Contabiliza telemetria
                motor_spike_counts += spk_motor
                ep_hidden_spikes += spk_hidden.sum().item()
                ep_motor_spikes += spk_motor.sum().item()
                ep_abs_dw1 += delta_w1.abs().sum().item()
                ep_abs_dw2 += delta_w2.abs().sum().item()

            # Normalização Homeostática ao final de cada passo para impedir explosão de pesos
            W1 = apply_synaptic_homeostasis(W1, target_row_mean=0.12)
            W2 = apply_synaptic_homeostasis(W2, target_row_mean=0.06)

            action_id = select_motor_action(motor_spike_counts, mem_motor)
            sensors, reward, done = world.step(action_id)
            episode_reward += reward
            steps += 1

            if done:
                if reward > 0:
                    total_goals_reached += 1
                break

        mean_spk_h = ep_hidden_spikes / max(1, steps)
        mean_spk_m = ep_motor_spikes / max(1, steps)

        print(
            f"Episodio {episode + 1:02d}/{num_episodes} | "
            f"Passos: {steps:03d} | "
            f"Metas: {total_goals_reached:02d} | "
            f"Spikes [H:{mean_spk_h:4.1f} M:{mean_spk_m:3.1f}] | "
            f"|ΔW| [W1:{ep_abs_dw1:.3f} W2:{ep_abs_dw2:.3f}]"
        )

    print("=" * 80)
    print(
        f"Experimento NeuroCore 0.1 com STDP concluído! "
        f"Taxa de Sucesso: {total_goals_reached}/{num_episodes} "
        f"({(total_goals_reached / num_episodes) * 100:.1f}%)"
    )


if __name__ == "__main__":
    run_stdp_synapse_test()
    print("=" * 80)
    run_neurocore_02_stdp_experiment()