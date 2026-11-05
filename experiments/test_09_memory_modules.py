'''Teste 09: retenção de um padrão em uma rede recorrente associativa.'''
import torch

from core.neuron import LeakyIntegrateAndFire


def run_memory_test():
    print("Iniciando Teste de Retenção da Memória Associativa (NeuroCore)...")
    torch.manual_seed(42)

    num_neurons = 100
    num_active = 20
    threshold = 1.0
    memory_layer = LeakyIntegrateAndFire(
        num_neurons=num_neurons,
        threshold=threshold,
        decay=0.95,
    )

    # Padrão determinístico de 20 bits, para tornar o teste reproduzível.
    input_pattern = torch.zeros(num_neurons, dtype=torch.float32)
    input_pattern[:num_active] = 1.0
    print(f"[Gravação] Padrão original: {int(input_pattern.sum().item())} neurônios ativos.")

    # Regra Hebbiana simples para armazenar este padrão. A normalização garante
    # corrente > threshold para cada neurônio ativo quando o padrão é reaplicado.
    recurrent_weights = torch.outer(input_pattern, input_pattern) * (1.1 / (num_active - 1))
    recurrent_weights.fill_diagonal_(0.0)

    recalled, _ = memory_layer(input_pattern * 2.0)
    assert torch.equal(recalled, input_pattern), "A codificação inicial não reproduziu o padrão apresentado."

    print("[Recuperação] Cinco ciclos recorrentes sem estímulo externo...")
    exact_matches = 0
    for step in range(5):
        recurrent_current = recurrent_weights @ recalled
        recalled, _ = memory_layer(recurrent_current)
        exact = torch.equal(recalled, input_pattern)
        exact_matches += int(exact)
        overlap = int(torch.logical_and(recalled.bool(), input_pattern.bool()).sum().item())
        false_positives = int(torch.logical_and(recalled.bool(), ~input_pattern.bool()).sum().item())
        print(
            f"  -> Ciclo {step + 1}: sobreposição={overlap}/{num_active}, "
            f"falsos positivos={false_positives}, padrão exato={'sim' if exact else 'não'}"
        )

    success = exact_matches == 5 and torch.equal(recalled, input_pattern)
    print("\n[Resultados - Teste 09]")
    print(f"Ciclos com recuperação exata: {exact_matches}/5")
    if not success:
        raise AssertionError("A rede não reteve e recuperou o padrão armazenado em todos os ciclos.")
    print("[PASS] O padrão armazenado foi mantido exatamente nos cinco ciclos de recuperação.")


if __name__ == "__main__":
    run_memory_test()
