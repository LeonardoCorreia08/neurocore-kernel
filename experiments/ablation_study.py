
"""
Módulo: experiments/ablation_study.py

ITEM 36 — Estudo de ablação científica.
Objetivo:
    Comparar uma rede com plasticidade habilitada com uma rede
    sem recompensas explícitas, medindo a proporção de spikes
    direcionados ao canal motor alvo.

Limitação:
    Este experimento depende da implementação real de SpikeEngine.
    A comparação, por si só, não prova que STDP/R-STDP seja a causa
    de qualquer diferença observada.
"""

import random
import statistics

import torch

from core.spike_engine import SpikeEngine


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SEED = 42

LAYER_SIZES = [16, 128, 4]
NETWORK_DENSITY = 0.3

TARGET_MOTOR_CHANNEL = 0

TRAINING_EPOCHS = 30
TRAINING_STEPS_PER_EPOCH = 5
EVALUATION_STEPS = 20

# Quantidade de execuções independentes para reduzir a influência
# de uma única inicialização aleatória.
REPEATS = 5

# Limita o crescimento do sinal de recompensa.
REWARD_MAGNITUDE = 1.0


def set_seed(seed: int) -> None:
    """Define as sementes dos geradores aleatórios disponíveis."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_input_pattern(device: torch.device) -> torch.Tensor:
    """Cria o padrão de entrada utilizado em todas as avaliações."""
    pattern = torch.zeros(16, device=device)
    pattern[[3, 7, 12]] = 4.0
    return pattern


def get_output_spikes(
    engine: SpikeEngine,
    input_pattern: torch.Tensor,
    device: torch.device,
    steps: int,
) -> torch.Tensor:
    """Avalia a saída reiniciando apenas os estados neuronais."""
    engine.reset_neuron_states()

    output_counts = torch.zeros(4, device="cpu")

    with torch.no_grad():
        for _ in range(steps):
            result = engine(input_pattern)
            output_counts += result[-1].detach().to("cpu").float()

    return output_counts


def calculate_metrics(
    output_counts: torch.Tensor,
    target_channel: int,
) -> dict:
    """Calcula a proporção de spikes no canal motor alvo."""
    total_spikes = output_counts.sum().item()
    target_spikes = output_counts[target_channel].item()

    if total_spikes > 0:
        target_ratio = target_spikes / total_spikes
    else:
        target_ratio = 0.0

    return {
        "distribution": output_counts.tolist(),
        "total_spikes": total_spikes,
        "target_spikes": target_spikes,
        "target_ratio": target_ratio,
        "accuracy_percent": target_ratio * 100.0,
    }


def calculate_reward(metrics: dict) -> float:
    """
    Produz recompensa positiva quando o alvo recebe spikes acima
    da distribuição uniforme esperada entre quatro canais.

    ATENÇÃO:
    Este sinal só terá o efeito pretendido se apply_global_reward()
    interpretar corretamente recompensas positivas e negativas.
    """
    ratio = metrics["target_ratio"]
    uniform_baseline = 1.0 / 4.0

    # Centraliza a recompensa em torno da distribuição uniforme.
    reward = (ratio - uniform_baseline) * 2.0

    # Limita o intervalo da recompensa a [-1, 1].
    return max(-REWARD_MAGNITUDE, min(REWARD_MAGNITUDE, reward))


def create_engine(
    device: torch.device,
    adaptive: bool = True,
) -> SpikeEngine:
    """Cria uma rede adaptativa ou uma rede de controle sem plasticidade."""
    return SpikeEngine(
        layer_sizes=LAYER_SIZES,
        density=NETWORK_DENSITY,
        device=str(device),
        enable_stdp=adaptive,
        enable_rstdp=adaptive,
        enable_homeostasis=adaptive,
    )


def train_adaptive_network(
    engine: SpikeEngine,
    input_pattern: torch.Tensor,
    device: torch.device,
) -> list:
    """
    Treina a rede adaptativa e registra a precisão observada por época.

    A recompensa é calculada a partir da resposta observada antes
    da atualização global de recompensa.
    """
    history = []

    for epoch in range(TRAINING_EPOCHS):
        # Fase de estímulo.
        for _ in range(TRAINING_STEPS_PER_EPOCH):
            engine(input_pattern)

        # Avalia a resposta da rede.
        output_counts = get_output_spikes(
            engine,
            input_pattern,
            device,
            EVALUATION_STEPS,
        )

        metrics = calculate_metrics(
            output_counts,
            TARGET_MOTOR_CHANNEL,
        )

        reward = calculate_reward(metrics)

        # Atualização fornecida pelo SpikeEngine.
        engine.apply_global_reward(reward=reward)

        history.append({
            "epoch": epoch + 1,
            "accuracy_percent": metrics["accuracy_percent"],
            "target_ratio": metrics["target_ratio"],
            "reward": reward,
            "distribution": metrics["distribution"],
        })

    return history


def train_static_network(
    engine: SpikeEngine,
    input_pattern: torch.Tensor,
) -> None:
    """
    Executa os mesmos estímulos sem aplicar recompensas explícitas.

    Isso não garante ausência total de aprendizagem: se SpikeEngine
    aplicar STDP automaticamente durante forward(), essa plasticidade
    ainda poderá ocorrer. A implementação precisa ser verificada.
    """
    for _ in range(TRAINING_EPOCHS):
        for _ in range(TRAINING_STEPS_PER_EPOCH):
            engine(input_pattern)


def evaluate_network(
    engine: SpikeEngine,
    input_pattern: torch.Tensor,
    device: torch.device,
) -> dict:
    """Executa a avaliação final da rede."""
    output_counts = get_output_spikes(
        engine,
        input_pattern,
        device,
        EVALUATION_STEPS,
    )

    return calculate_metrics(output_counts, TARGET_MOTOR_CHANNEL)


def print_result(label: str, result: dict) -> None:
    """Imprime as métricas de uma rede."""
    print(f"\n{label}")
    print(f"  Distribuição de spikes : {result['distribution']}")
    print(f"  Spikes totais          : {result['total_spikes']:.0f}")
    print(f"  Spikes no alvo         : {result['target_spikes']:.0f}")
    print(f"  Proporção no alvo      : {result['accuracy_percent']:.2f}%")


def run_ablation_study() -> None:
    print("=" * 75)
    print("ITEM 36: ESTUDO DE ABLAÇÃO — PRECISÃO DIRECIONAL")
    print("=" * 75)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo: {device}")
    print(f"Semente base: {SEED}")
    print(f"Repetições: {REPEATS}")
    print(f"Canal motor alvo: {TARGET_MOTOR_CHANNEL}")
    print(f"Arquitetura: {LAYER_SIZES}")
    print(f"Densidade: {NETWORK_DENSITY}")

    full_results = []
    static_results = []
    all_histories = []

    for repeat in range(REPEATS):
        print(f"\n{'-' * 75}")
        print(f"EXECUÇÃO {repeat + 1}/{REPEATS}")
        print("-" * 75)

        # Cada repetição tem uma semente determinística.
        repeat_seed = SEED + repeat
        set_seed(repeat_seed)

        input_pattern = make_input_pattern(device)

        # Rede adaptativa.
        print("\n[Cenário 1] Rede com recompensa global")
        engine_full = create_engine(device, adaptive=True)
        # Rede de controle sem plasticidade
        engine_static = create_engine(device, adaptive=False)
        

        history = train_adaptive_network(
            engine_full,
            input_pattern,
            device,
        )

        full_result = evaluate_network(
            engine_full,
            input_pattern,
            device,
        )

        full_results.append(full_result)
        all_histories.append(history)

        print_result("Resultado adaptativo", full_result)

        # Rede sem recompensas explícitas.
        #
        # A semente é reiniciada para tentar igualar as condições
        # de inicialização. A equivalência real dos pesos depende
        # do consumo de números aleatórios pelo SpikeEngine.
        set_seed(repeat_seed)

        print("\n[Cenário 2] Rede sem recompensas explícitas")
        engine_static = create_engine(device)

        train_static_network(
            engine_static,
            input_pattern,
        )

        static_result = evaluate_network(
            engine_static,
            input_pattern,
            device,
        )

        static_results.append(static_result)

        print_result("Resultado sem recompensa", static_result)

    # ========================================================
    # RESUMO AGREGADO
    # ========================================================

    adaptive_scores = [
        result["accuracy_percent"] for result in full_results
    ]
    static_scores = [
        result["accuracy_percent"] for result in static_results
    ]

    adaptive_mean = statistics.mean(adaptive_scores)
    static_mean = statistics.mean(static_scores)

    adaptive_std = (
        statistics.stdev(adaptive_scores)
        if len(adaptive_scores) > 1 else 0.0
    )
    static_std = (
        statistics.stdev(static_scores)
        if len(static_scores) > 1 else 0.0
    )

    difference = adaptive_mean - static_mean

    print("\n" + "=" * 75)
    print("RELATÓRIO FINAL DE ABLAÇÃO")
    print("=" * 75)

    print(f"Alvo motor: canal {TARGET_MOTOR_CHANNEL}")
    print(f"Passos de avaliação por execução: {EVALUATION_STEPS}")
    print(f"Épocas de treinamento: {TRAINING_EPOCHS}")
    print(f"Repetições: {REPEATS}")

    print("\nPrecisão direcional média:")
    print(
        f"  Rede adaptativa       : "
        f"{adaptive_mean:.2f}% ± {adaptive_std:.2f}%"
    )
    print(
        f"  Sem recompensa global: "
        f"{static_mean:.2f}% ± {static_std:.2f}%"
    )
    print(f"  Diferença média       : {difference:+.2f} pontos percentuais")

    print("\nResultados por repetição:")
    for index, (adaptive, static) in enumerate(
        zip(adaptive_scores, static_scores),
        start=1,
    ):
        print(
            f"  {index}. Adaptativa={adaptive:.2f}% | "
            f"Sem recompensa={static:.2f}%"
        )

    if difference > 0:
        print(
            "\nConclusão descritiva: a rede adaptativa apresentou "
            "maior proporção média de spikes no alvo nesta amostra."
        )
    elif difference < 0:
        print(
            "\nConclusão descritiva: a rede sem recompensa explícita "
            "apresentou maior proporção média de spikes no alvo nesta amostra."
        )
    else:
        print(
            "\nConclusão descritiva: as duas condições apresentaram "
            "a mesma proporção média no alvo nesta amostra."
        )

    print(
        "\nLimitação: esta comparação não comprova causalidade nem "
        "demonstra que a rede estática não possui plasticidade. "
        "Verifique a implementação de SpikeEngine, STDP e "
        "apply_global_reward() antes de interpretar o resultado."
    )
    print("=" * 75)


if __name__ == "__main__":
    run_ablation_study()
