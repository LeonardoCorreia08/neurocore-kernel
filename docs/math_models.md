# Modelos matemáticos do NEUROCORE

## 1. Escopo

Este documento regista os modelos matemáticos descritos no projeto NEUROCORE, atualizados para refletir com exatidão a implementação real do código baseada em tensores esparsos (PyTorch).

Os modelos centrais são:

- neurónio Leaky Integrate-and-Fire (LIF);
- sinapse com propagação esparsa instantânea;
- plasticidade dependente do tempo de disparo (STDP);
- plasticidade modulada por recompensa (R-STDP);
- regulação homeostática;
- métricas de atividade e esparsidade.

## 2. Modelo neuronal LIF

O modelo LIF representa a integração de corrente na membrana de um neurónio. O potencial diminui ao longo do tempo por causa do vazamento.

A equação diferencial clássica é:

\[
\tau_m \frac{dV(t)}{dt}
=
-(V(t)-V_{rest}) + R I(t)
\]

### 2.1 Regra de disparo

O neurónio dispara quando o potencial atinge ou ultrapassa um limiar:

\[
V(t) \geq V_{th}
\]

O evento de disparo é representado por um pulso binário (*spike*, 1 ou 0). Após o disparo, o modelo reinicia o potencial para o estado de repouso (reset).

### 2.2 Forma discreta (Implementação Tensorial)

Em vez da integração de Euler genérica, a implementação na classe `LeakyIntegrateAndFire` (`core/neuron.py`) adota uma forma discreta otimizada para operações tensoriais em lote, onde o decaimento é aplicado linearmente:

\[
V_t = V_{t-1} \times \beta + I_t
\]

Onde:
- \(\beta\) é o fator de retenção (decaimento ou *leak*) fixado internamente na classe.
- \(I_t\) é o tensor de corrente recebido da camada sináptica no passo de tempo atual.

## 3. Corrente sináptica

A arquitetura do NeuroCore prioriza a eficiência, utilizando a propagação de estímulos sem atrasos temporais individuais, o que permite o uso de multiplicação de matrizes ultra-otimizadas.

A entrada global de uma camada pós-sináptica é calculada matricialmente como:

\[
I(t) = W_{esparsa} \cdot S_{pre}(t)
\]

Onde:
- \(I(t)\) é a corrente global recebida no tempo \(t\).
- \(W_{esparsa}\) é a matriz de pesos sinápticos formatada como `torch.sparse_coo_tensor`.
- \(S_{pre}(t)\) é o vetor de disparos (*spikes*) da camada pré-sináptica.

A implementação atual utiliza o método `torch.sparse.mm` para maximizar o desempenho e evitar o esgotamento de memória, mesmo na escala de 1 milhão de neurónios.

## 4. Plasticidade dependente do tempo de disparo (STDP)

A STDP altera o peso de uma sinapse com base na diferença entre os tempos de disparo pré-sináptico e pós-sináptico. A implementação matricial calcula as aproximações em tensores.

A diferença conceitual é definida como:
\[
\Delta t = t_{pos} - t_{pre}
\]

A regra exponencial teórica é:
\[
\Delta w =
\begin{cases}
A_+ e^{-\Delta t/\tau_+}, & \Delta t > 0\\
-A_- e^{\Delta t/\tau_-}, & \Delta t < 0
\end{cases}
\]

Onde:
- \(\Delta w\) é a alteração do peso sináptico.
- \(A_+\), \(A_-\) são as amplitudes de fortalecimento e enfraquecimento.
- \(\tau_+\), \(\tau_-\) são as constantes temporais.

### 4.2 Limites do peso (Clamping)

A implementação da classe `SparseSynapseManager` restringe estritamente as atualizações de peso para evitar que as sinapses saturem ao infinito. Isto é garantido de forma explícita através da operação `torch.clamp`:

\[
-1.0 \leq w_{ij} \leq 1.0
\]

Pesos iniciais são distribuídos em torno de zero, controlados por um fator de escala padrão (*weight_scale*) de \(0.2\).

## 5. STDP modulada por recompensa (R-STDP)

O projeto descreve a modulação da plasticidade por um sinal de recompensa global. A ideia é usar o resultado de uma ação para modular as alterações sinápticas.

Uma representação conceitual é:
\[
\Delta w_{ij} = M(t)\,e_{ij}(t)
\]

Onde:
- \(M(t)\) é o sinal modulador de recompensa.
- \(e_{ij}(t)\) é o traço de elegibilidade associado à atividade da sinapse.

## 6. Plasticidade homeostática

A plasticidade homeostática busca manter a atividade neural em uma faixa estável. 

Uma representação conceitual de uma atualização de limiar é:
\[
\theta_i(t+\Delta t)
=
\theta_i(t) + \eta_h\left(r_i(t)-r_{target}\right)
\]

Onde \(\theta_i\) é o limiar, \(\eta_h\) é a taxa de adaptação, \(r_i\) é a taxa observada e \(r_{target}\) é a taxa desejada.

## 7. Codificação de estímulos

O projeto usa codificadores para converter valores contínuos em tensores binários (*spikes*).
- **Codificação por taxa:** O estímulo é proporcional à frequência de disparos (\(r = N_{spikes}/T\)).
- **Codificação temporal:** A representação do sinal dá-se pela posição exata dos tensores ativados no passo de tempo.

## 8. Conectividade esparsa

O projeto usa o formato de coordenadas esparsas (COO) do PyTorch. A densidade real de conexões é descrita por:

\[
\rho = \frac{N_{conexões}}{N_{neurónios\_pre} \times N_{neurónios\_pos}}
\]

Os testes de carga validaram que o `SparseSynapseManager` sustenta simulações contínuas ajustando a densidade \(\rho\) com base na escala, tipicamente a:
- 15% (0.15) para operações padrão de 10.000 neurónios.
- 10% (0.10) para 100.000 neurónios.
- 1% (0.01) até 10% (0.10) para testes extremos de 1 milhão de neurónios.

## 9. Métricas registradas

### 9.1 Taxa de disparo
\[
r = \frac{N_{spikes}}{N_{neurónios}}
\]
Calculada estaticamente a cada passo (*step*), conforme os testes dinâmicos registam (ex: média de 230 mil disparos/passo em 1 milhão de neurónios, refletindo uma rede ativa a ~23%).

### 9.2 Taxa de sucesso (Generalização / R-STDP)
A taxa de transferência em testes *zero-shot* é validada computando episódios em que a meta ou o limiar-alvo é atingido sem necessidade de retreino:
\[
P_{sucesso} = \frac{N_{episódios\ com\ meta\ atingida}}{N_{episódios\ totais}}\times 100
\]

## 10. Parâmetros base da implementação

Tabela correspondente à arquitetura validada nos testes experimentais:

| Parâmetro | Símbolo | Estado da documentação no Código Real |
|---|---|---|
| Fator de decaimento (Leak) | \(\beta\) | Gerido de forma encapsulada no `LeakyIntegrateAndFire`. |
| Limiar de disparo | \(V_{th}\) | Integrado à função de ativação do neurónio. |
| Fator Escala de Peso | \(W_{scale}\) | 0.2 (aplicado na inicialização `torch.randn`). |
| Limite Mínimo de Peso | \(w_{\min}\) | -1.0 (garantido através de `torch.clamp`). |
| Limite Máximo de Peso | \(w_{\max}\) | 1.0 (garantido através de `torch.clamp`). |
| Densidade Default | \(\rho\) | 0.15 a 0.10, suportando redução escalável. |

## 11. Referências de implementação

O repositório distribui os cálculos nos seguintes módulos:
- `core/neuron.py` (LIF dinâmico e retenção)
- `core/synapse.py` (Multiplicação tensorial esparsa e clamps de peso)
- `experiments/` (Scripts de validação matemática e de telemetria de longo prazo)