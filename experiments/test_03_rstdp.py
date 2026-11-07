'''Modulo: test_03_rstdp.py (NEUROCORE v3.2 - Auditado)'''
import torch
import numpy as np

from core.neuron import LeakyIntegrateAndFire
from environment.world import VirtualWorld2D
from learning.stdp import STDPManager
from learning.reward_learning import RSTDPManager


# ============================================================
# CONFIGURAÇÕES RÁPIDAS (PERFIL EXPERIMENTAL EXPLÍCITO)
# ============================================================

SEED = 42

INPUT_DIM = 4
HIDDEN_DIM = 16
OUTPUT_DIM = 4

SNN_TICKS = 5
MAX_STEPS = 200
NUM_EPISODES = 50

CORTICAL_NOISE = 0.08          # Reduzido de 0.12 para estabilizar a política após aprender

W1_TARGET_MEAN = 0.14          # Mantém excitação saudável para evitar passos "Sem Spike"
W2_INIT_MEAN = 0.08

W_MIN = -1.0
W_MAX = 1.0

STDP_W1_LR = 0.005
RSTDP_W2_LR = 0.015            # Alinhado para evitar saturação abrupta

HOMEOSTASIS_INTERVAL = 10

EPSILON_INITIAL = 0.20
EPSILON_MIN = 0.02

DEVICE = torch.device("cpu")


# ============================================================
# 1. HOMEOSTASE SINÁPTICA EXATA (CORRIGIDA PÓS-CLAMP)
# ============================================================

def apply_synaptic_homeostasis(
    W: torch.Tensor,
    target_row_mean: float,
    w_min: float = W_MIN,
    w_max: float = W_MAX,
    iterations: int = 3
) -> torch.Tensor:
    """
    Normaliza a média por linha (neurônio pós-sináptico) usando projeção
    iterativa para garantir que o clamp não distorça a média final solicitada.
    """
    W_out = torch.clamp(W, w_min, w_max)
    for _ in range(iterations):
        current_mean = W_out.mean(dim=1, keepdim=True)
        W_out = W_out - current_mean + target_row_mean
        W_out = torch.clamp(W_out, w_min, w_max)
    return W_out


# ============================================================
# 2. ÍNDICES SINÁPTICOS
# ============================================================

def dense_sparse_indices(num_posts: int, num_pres: int) -> torch.Tensor:
    post_idx, pre_idx = torch.meshgrid(
        torch.arange(num_posts, device=DEVICE),
        torch.arange(num_pres, device=DEVICE),
        indexing="ij"
    )
    return torch.stack([
        post_idx.reshape(-1),
        pre_idx.reshape(-1)
    ])


# ============================================================
# 3. DECODIFICADOR MOTOR INSTRUMENTADO
# ============================================================

def select_motor_action(
    motor_spike_counts: torch.Tensor,
    mem_motor_trace: torch.Tensor,
    epsilon: float = 0.05
) -> tuple[int, str]:
    """
    Decodifica a ação motora usando:
    1. Exploração epsilon-greedy controlada
    2. Contagem de spikes na janela temporal (SNN_TICKS)
    3. Desempate/Fallback pelo traço médio de membrana na janela (e não apenas o último tick)
    Retorna (action_id, modo_de_decisao).
    """
    if np.random.random() < epsilon:
        return int(np.random.randint(0, len(mem_motor_trace))), "explore"

    max_spikes = motor_spike_counts.max().item()

    if max_spikes > 0:
        winners = torch.nonzero(motor_spike_counts == max_spikes).flatten()
        if winners.numel() == 1:
            return int(winners.item()), "spike"
        winner_mem = mem_motor_trace[winners]
        return int(winners[winner_mem.argmax()].item()), "spike_tie"

    return int(mem_motor_trace.argmax().item()), "membrane_fallback"


# ============================================================
# 4. EXTRAÇÃO DE DISTÂNCIA E RECOMPENSA DE PROGRESSO ATIVA
# ============================================================

def get_environment_distance(world: VirtualWorld2D, sensors) -> float:
    """
    Inspeciona a instância real de VirtualWorld2D para extrair a distância
    euclidiana até a meta. Caso os atributos tenham outros nomes, utiliza
    potencial sensorial contínuo como proxy livre de invenção de coordenadas.
    """
    # Tentativa 1: Atributos diretos de posição do agente e meta no VirtualWorld2D
    pos_attrs = [("agent_pos", "goal_pos"), ("pos", "goal"), ("agent", "target"), ("x", "goal_x")]
    for a_attr, g_attr in pos_attrs:
        if hasattr(world, a_attr) and hasattr(world, g_attr):
            a_val = np.asarray(getattr(world, a_attr), dtype=np.float32)
            g_val = np.asarray(getattr(world, g_attr), dtype=np.float32)
            if a_val.shape == g_val.shape and a_val.size >= 2:
                return float(np.linalg.norm(g_val - a_val))

    # Tentativa 2: Atributos escalares x/y
    if all(hasattr(world, attr) for attr in ("agent_x", "agent_y", "goal_x", "goal_y")):
        dx = float(world.goal_x - world.agent_x)
        dy = float(world.goal_y - world.agent_y)
        return float(np.hypot(dx, dy))

    # Tentativa 3: Proxy diretamente derivado da intensidade sensorial (1 / (1 + max_sensor))
    s_arr = np.asarray(sensors, dtype=np.float32)
    return float(1.0 / (1.0 + np.max(np.abs(s_arr))))


def calculate_dopamine(
    reward: float,
    previous_distance: float | None,
    current_distance: float | None
) -> float:
    """
    Combina reforço terminal com Potential-Based Reward Shaping (PBRS).
    """
    if reward > 0.1:
        return 2.0
    if reward < -0.1:
        return -0.25

    if previous_distance is not None and current_distance is not None:
        progress = previous_distance - current_distance
        return float(np.clip(0.08 * progress, -0.04, 0.04))

    return 0.0


# ============================================================
# 5. TESTE UNITÁRIO R-STDP (COBERTURA COMPLETA DA AUDITORIA)
# ============================================================

def run_rstdp_synapse_test():
    print("\nTESTE UNITÁRIO R-STDP (AUDITORIA MATEMÁTICA COMPLETA)")
    torch.manual_seed(SEED)

    lr_test = 0.1
    rstdp = RSTDPManager(num_posts=1, num_pres=1, lr=lr_test)
    indices = torch.tensor([[0], [0]], dtype=torch.long)

    # 1. Ausência de atividade -> Elegibilidade zero
    rstdp.reset_state()
    rstdp.step_eligibility(torch.tensor([0.0]), torch.tensor([0.0]), indices)
    dw_idle = rstdp.apply_reward_modulation(1.0, indices).item()
    assert abs(dw_idle) < 1e-8, f"Esperado 0 sem atividade, obteve {dw_idle}"

    # 2. Ordem Causal (Pré -> Pós) + Recompensa Positiva (+1.0)
    rstdp.reset_state()
    rstdp.step_eligibility(torch.tensor([1.0]), torch.tensor([0.0]), indices)
    rstdp.step_eligibility(torch.tensor([0.0]), torch.tensor([1.0]), indices)
    dw_positive = rstdp.apply_reward_modulation(1.0, indices).item()
    assert dw_positive > 0

    # 3. Ordem Causal (Pré -> Pós) + Recompensa Zero (0.0) -> Delta W = 0
    rstdp.reset_state()
    rstdp.step_eligibility(torch.tensor([1.0]), torch.tensor([0.0]), indices)
    rstdp.step_eligibility(torch.tensor([0.0]), torch.tensor([1.0]), indices)
    dw_zero_rew = rstdp.apply_reward_modulation(0.0, indices).item()
    assert abs(dw_zero_rew) < 1e-8, f"Recompensa 0 deve gerar dw=0, obteve {dw_zero_rew}"

    # 4. Ordem Causal (Pré -> Pós) + Punição (-1.0) -> Simetria exata
    rstdp.reset_state()
    rstdp.step_eligibility(torch.tensor([1.0]), torch.tensor([0.0]), indices)
    rstdp.step_eligibility(torch.tensor([0.0]), torch.tensor([1.0]), indices)
    dw_negative = rstdp.apply_reward_modulation(-1.0, indices).item()
    assert dw_negative < 0
    assert abs(dw_positive + dw_negative) < 1e-6

    # 5. Ordem Anti-Causal / Inversa (Pós -> Pré) -> Elegibilidade Negativa ou Nula
    rstdp.reset_state()
    rstdp.step_eligibility(torch.tensor([0.0]), torch.tensor([1.0]), indices)
    rstdp.step_eligibility(torch.tensor([1.0]), torch.tensor([0.0]), indices)
    dw_anticausal = rstdp.apply_reward_modulation(1.0, indices).item()
    assert dw_anticausal <= 0.0, f"Ordem Pós->Pré não pode gerar LTP positivo: {dw_anticausal}"

    # 6. Curva de Decaimento Temporal em múltiplos intervalos [0, 1, 5, 10, 25, 50]
    decay_steps = [0, 1, 5, 10, 25, 50]
    decay_values = []
    for k in decay_steps:
        rstdp.reset_state()
        rstdp.step_eligibility(torch.tensor([1.0]), torch.tensor([0.0]), indices)
        rstdp.step_eligibility(torch.tensor([0.0]), torch.tensor([1.0]), indices)
        for _ in range(k):
            rstdp.step_eligibility(torch.tensor([0.0]), torch.tensor([0.0]), indices)
        decay_values.append(rstdp.apply_reward_modulation(1.0, indices).item())

    for i in range(len(decay_values) - 1):
        assert decay_values[i] > decay_values[i + 1] > 0, "Curva de decaimento não é estritamente monotônica!"

    # Verifica constante de tempo tau_eligibility (5 passos: razão e^(-5/50) = 0.904837)
    ratio_5_steps = decay_values[2] / decay_values[0]
    expected_ratio = float(np.exp(-5.0 / 50.0))
    assert abs(ratio_5_steps - expected_ratio) < 1e-3, (
        f"Decaimento descalibrado: razão={ratio_5_steps:.6f} vs esperado={expected_ratio:.6f}"
    )

    print(f"Reforço positivo (0 passos): {decay_values[0]:+.6f}")
    print(f"Punição (-1.0):              {dw_negative:+.6f}")
    print(f"Ordem inversa (Pós->Pré):    {dw_anticausal:+.6f}")
    print(f"Curva Decaimento [0,1,5,10,25,50]: {[round(v, 5) for v in decay_values]}")
    print("[PASS] Todos os 6 invariantes matemáticos do R-STDP validados!")


# ============================================================
# 6. EXPERIMENTO NEUROCORE v3.2 (AUDITADO)
# ============================================================

def run_neurocore_v32():
    print("\n" + "=" * 95)
    print("NEUROCORE v3.2 - R-STDP COM ELEGIBILIDADE CONDICIONADA À AÇÃO & SHAPING ATIVO")
    print("=" * 95)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    world = VirtualWorld2D()

    sensory_layer = LeakyIntegrateAndFire(num_neurons=INPUT_DIM)
    hidden_layer = LeakyIntegrateAndFire(num_neurons=HIDDEN_DIM)
    motor_layer = LeakyIntegrateAndFire(num_neurons=OUTPUT_DIM)

    # Inicialização com Homeostase Exata
    W1 = torch.randn(HIDDEN_DIM, INPUT_DIM) / np.sqrt(INPUT_DIM)
    W1 = apply_synaptic_homeostasis(W1, W1_TARGET_MEAN)

    W2 = torch.randn(OUTPUT_DIM, HIDDEN_DIM) / np.sqrt(HIDDEN_DIM)
    W2 = apply_synaptic_homeostasis(W2, W2_INIT_MEAN)

    indices_w1 = dense_sparse_indices(HIDDEN_DIM, INPUT_DIM)
    indices_w2 = dense_sparse_indices(OUTPUT_DIM, HIDDEN_DIM)

    stdp_w1 = STDPManager(
        num_posts=HIDDEN_DIM,
        num_pres=INPUT_DIM,
        lr=STDP_W1_LR
    )

    rstdp_w2 = RSTDPManager(
        num_posts=OUTPUT_DIM,
        num_pres=HIDDEN_DIM,
        lr=RSTDP_W2_LR
    )

    total_goals = 0
    total_collisions = 0
    total_timeouts = 0
    success_history = []

    for episode in range(NUM_EPISODES):
        sensors = world.reset()

        sensory_layer.reset_state()
        hidden_layer.reset_state()
        motor_layer.reset_state()

        stdp_w1.reset_traces()
        rstdp_w2.reset_state()

        motor_fatigue = torch.zeros(OUTPUT_DIM)

        steps = 0
        episode_goals = 0
        episode_collisions = 0
        hidden_spikes = 0.0
        motor_spikes = 0.0
        dopamine_total = 0.0
        no_spike_decisions = 0

        # Distância inicial real (ativa calculate_dopamine)
        previous_distance = get_environment_distance(world, sensors)

        while steps < MAX_STEPS:
            x_sensory = torch.as_tensor(sensors, dtype=torch.float32)

            motor_spike_counts = torch.zeros(OUTPUT_DIM)
            mem_motor_accum = torch.zeros(OUTPUT_DIM)
            hidden_spike_accum = torch.zeros(HIDDEN_DIM)

            # ------------------------------------------------
            # DINÂMICA SNN (MICRO-TICKS)
            # ------------------------------------------------
            for _ in range(SNN_TICKS):
                sensory_drive = x_sensory + torch.randn(INPUT_DIM) * 0.04
                spk_sensory, _ = sensory_layer(sensory_drive)

                h_input = (
                    torch.matmul(W1, spk_sensory)
                    + torch.randn(HIDDEN_DIM) * CORTICAL_NOISE
                )
                spk_hidden, _ = hidden_layer(h_input)

                m_input = (
                    torch.matmul(W2, spk_hidden)
                    - motor_fatigue
                    + torch.randn(OUTPUT_DIM) * CORTICAL_NOISE
                )
                spk_motor, mem_motor = motor_layer(m_input)

                motor_fatigue = motor_fatigue * 0.92 + spk_motor * 0.12

                hidden_layer.mem.clamp_(min=-0.5)
                motor_layer.mem.clamp_(min=-0.5)

                # STDP em W1 com CLAMP IMEDIATO (Correção Ponto 4 da Auditoria)
                delta_w1 = stdp_w1.compute_sparse_deltas(
                    spk_sensory,
                    spk_hidden,
                    indices_w1
                ).view_as(W1)
                W1 = torch.clamp(W1 + delta_w1, W_MIN, W_MAX)

                motor_spike_counts += spk_motor
                mem_motor_accum += mem_motor
                hidden_spike_accum += spk_hidden

                hidden_spikes += spk_hidden.sum().item()
                motor_spikes += spk_motor.sum().item()

            # Homeostase periódica APENAS em W1 (preserva política aprendida em W2!)
            if (steps + 1) % HOMEOSTASIS_INTERVAL == 0:
                W1 = apply_synaptic_homeostasis(W1, W1_TARGET_MEAN)

            # ------------------------------------------------
            # DECISÃO MOTORA
            # ------------------------------------------------
            if motor_spike_counts.sum() == 0:
                no_spike_decisions += 1

            epsilon = max(
                EPSILON_MIN,
                EPSILON_INITIAL * (1.0 - episode / NUM_EPISODES)
            )

            mem_motor_mean = mem_motor_accum / SNN_TICKS
            action_id, _ = select_motor_action(
                motor_spike_counts,
                mem_motor_mean,
                epsilon
            )

            # ------------------------------------------------
            # ELEGIBILIDADE CONDICIONADA À AÇÃO EXECUTADA (CORREÇÃO CRÍTICA)
            # Garante que os neurônios ocultos ativos neste passo criem traço causal
            # exclusivamente com a ação motora que foi de fato enviada ao ambiente!
            # ------------------------------------------------
            active_hidden = (hidden_spike_accum > 0).float()
            if active_hidden.sum() == 0:
                # Se até a camada oculta calou, usa os neurônios mais despolarizados
                top_h = torch.argmax(hidden_layer.mem)
                active_hidden[top_h] = 1.0

            chosen_motor_spike = torch.zeros(OUTPUT_DIM)
            chosen_motor_spike[action_id] = 1.0

            # Sequência Pré (Oculto) -> Pós (Ação Escolhida) no gerenciador de elegibilidade
            rstdp_w2.step_eligibility(active_hidden, torch.zeros(OUTPUT_DIM), indices_w2)
            rstdp_w2.step_eligibility(torch.zeros(HIDDEN_DIM), chosen_motor_spike, indices_w2)

            # ------------------------------------------------
            # PASSO NO AMBIENTE E CÁLCULO DE DOPAMINA ATIVO
            # ------------------------------------------------
            next_sensors, reward, done = world.step(action_id)
            steps += 1

            current_distance = get_environment_distance(world, next_sensors)
            dopamine = calculate_dopamine(reward, previous_distance, current_distance)
            previous_distance = current_distance

            # Penalidade leve de estagnação se estourar MAX_STEPS sem achar a meta
            if steps == MAX_STEPS and not done and reward <= 0.1:
                dopamine = -0.10

            # ------------------------------------------------
            # MODULAÇÃO DOPAMINÉRGICA EM W2
            # ------------------------------------------------
            if abs(dopamine) > 1e-6:
                delta_w2 = rstdp_w2.apply_reward_modulation(
                    dopamine,
                    indices_w2
                ).view_as(W2)

                W2 = torch.clamp(W2 + delta_w2, min=W_MIN, max=W_MAX)
                dopamine_total += dopamine

                # Em eventos terminais (meta ou colisão), limpa o traço para não vazar
                if reward > 0.1 or reward < -0.1:
                    rstdp_w2.reset_state()

            if reward > 0.1:
                episode_goals += 1
                total_goals += 1
            elif reward < -0.1:
                episode_collisions += 1
                total_collisions += 1

            sensors = next_sensors

            if done:
                break

        if episode_goals == 0 and episode_collisions == 0:
            total_timeouts += 1

        # ----------------------------------------------------
        # TELEMETRIA POR EPISÓDIO (TAXA POR NEURÔNIO E POR PASSO)
        # ----------------------------------------------------
        success = 1 if episode_goals > 0 else 0
        success_history.append(success)
        recent_success = np.mean(success_history[-10:]) * 100

        mean_hidden = hidden_spikes / max(steps, 1)
        mean_motor = motor_spikes / max(steps, 1)

        print(
            f"Ep {episode + 1:03d}/{NUM_EPISODES} | "
            f"Passos: {steps:03d} | "
            f"Metas: {episode_goals} | "
            f"Colisões: {episode_collisions:02d} | "
            f"Spikes H:{mean_hidden:5.2f} | "
            f"M:{mean_motor:5.2f} | "
            f"Dopa:{dopamine_total:+6.2f} | "
            f"Sem Spike:{no_spike_decisions:03d} | "
            f"Sucesso10:{recent_success:5.1f}%"
        )

    # --------------------------------------------------------
    # RELATÓRIO DE AUDITORIA FINAL
    # --------------------------------------------------------
    w1_sat = ((W1.abs() >= 0.99).float().mean().item()) * 100
    w2_sat = ((W2.abs() >= 0.99).float().mean().item()) * 100
    first_half_acc = np.mean(success_history[:25]) * 100
    second_half_acc = np.mean(success_history[25:]) * 100

    print("\n" + "=" * 95)
    print("NEUROCORE v3.2 FINALIZADA - RELATÓRIO DE AUDITORIA")
    print("=" * 95)
    print(f"Metas Alcançadas:     {total_goals}/{NUM_EPISODES} ({np.mean(success_history) * 100:.2f}%)")
    print(f"Colisões Totais:      {total_collisions}/{NUM_EPISODES} ({(total_collisions / NUM_EPISODES) * 100:.2f}%)")
    print(f"Timeouts (Limite):    {total_timeouts}/{NUM_EPISODES} ({(total_timeouts / NUM_EPISODES) * 100:.2f}%)")
    print(f"Evolução Aprendizado: 1ª Metade (Ep 1-25): {first_half_acc:.1f}%  -->  2ª Metade (Ep 26-50): {second_half_acc:.1f}%")
    print(f"Camada W1 (STDP):     Média={W1.mean().item():+.5f} | Std={W1.std().item():.5f} | Saturados={w1_sat:.1f}%")
    print(f"Camada W2 (R-STDP):   Média={W2.mean().item():+.5f} | Std={W2.std().item():.5f} | Saturados={w2_sat:.1f}%")
    print("=" * 95)


if __name__ == "__main__":
    run_rstdp_synapse_test()
    run_neurocore_v32()