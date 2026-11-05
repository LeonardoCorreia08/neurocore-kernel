'''Teste 15: resposta neural sob mudança de domínio (não mede sucesso em tarefa).'''
import math
import torch

from core.neuron import LeakyIntegrateAndFire
from core.synapse import SparseSynapseManager
from learning.stdp import STDPManager
from monitoring.experiment_logger import ExperimentLogger


def _input_signal(input_dim: int, shifted: bool) -> torch.Tensor:
    """Gera estímulos binários no domínio A ou com ruído/deslocamento no domínio B."""
    if not shifted:
        return (torch.rand(input_dim) > 0.8).float()
    noise = torch.rand(input_dim) * 0.4
    return ((torch.rand(input_dim) > 0.75).float() + noise).clamp(0.0, 1.0)


def _measure_activity(sensory, hidden, synapse, episodes: int, steps: int, shifted: bool) -> float:
    rates = []
    for _ in range(episodes):
        sensory.reset_state()
        hidden.reset_state()
        episode_rates = []
        for _ in range(steps):
            signal = _input_signal(sensory.num_neurons, shifted)
            spikes_in, _ = sensory(signal)
            current = synapse.forward_propagation(spikes_in)
            spikes_hidden, _ = hidden(current)
            episode_rates.append(float(spikes_hidden.float().mean().item()))
        rates.append(sum(episode_rates) / len(episode_rates))
    return sum(rates) / len(rates)


def run_generalization_test():
    print("Iniciando Teste 15: atividade neural sob mudança de domínio...")

    input_dim = 100
    hidden_dim = 500
    episodes_train = 20
    episodes_eval = 20
    steps_per_episode = 200
    density = 0.1
    seed = 42

    ExperimentLogger.set_seed(seed)
    params = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "density": density,
        "episodes_train": episodes_train,
        "episodes_eval": episodes_eval,
        "steps_per_episode": steps_per_episode,
        "measurement": "hidden firing fraction; not task success",
    }
    ExperimentLogger.save_experiment_log(
        experiment_name="test_15_domain_shift_activity",
        params=params,
        seed=seed,
    )

    sensory = LeakyIntegrateAndFire(num_neurons=input_dim)
    hidden = LeakyIntegrateAndFire(num_neurons=hidden_dim)
    synapse = SparseSynapseManager(
        num_posts=hidden_dim,
        num_pres=input_dim,
        density=density,
    )
    stdp = STDPManager(num_posts=hidden_dim, num_pres=input_dim)
    stdp.reset_traces()

    print("[Treino] Exposição STDP não supervisionada a estímulos do domínio A...")
    for _ in range(episodes_train):
        sensory.reset_state()
        hidden.reset_state()
        for _ in range(steps_per_episode):
            signal = _input_signal(input_dim, shifted=False)
            spikes_in, _ = sensory(signal)
            current = synapse.forward_propagation(spikes_in)
            spikes_hidden, _ = hidden(current)
            delta = stdp.compute_sparse_deltas(
                spikes_in,
                spikes_hidden,
                synapse.sparse_weights.coalesce().indices(),
            )
            synapse.update_sparse_weights(delta)

    print("[Avaliação] Pesos congelados; comparando os domínios A e B...")
    activity_a = _measure_activity(
        sensory, hidden, synapse, episodes_eval, steps_per_episode, shifted=False
    )
    activity_b = _measure_activity(
        sensory, hidden, synapse, episodes_eval, steps_per_episode, shifted=True
    )
    if not (math.isfinite(activity_a) and math.isfinite(activity_b)):
        raise RuntimeError("A atividade neural avaliada contém valor não finito.")

    absolute_shift = abs(activity_b - activity_a)
    print(f"[Métrica A] Fração média de neurônios ocultos ativos por passo: {activity_a:.6f}")
    print(f"[Métrica B] Fração média de neurônios ocultos ativos por passo: {activity_b:.6f}")
    print(f"[Diferença absoluta de atividade] {absolute_shift:.6f}")
    print(
        "O teste mede resposta neural a uma mudança de distribuição; "
        "não mede alcance de metas nem comprova generalização de tarefa."
    )


if __name__ == "__main__":
    run_generalization_test()
