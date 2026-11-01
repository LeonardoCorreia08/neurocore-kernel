# Modelos matemáticos do NEUROCORE

## 1. Escopo

Este documento registra os modelos matemáticos descritos no README do NEUROCORE. Ele também identifica os pontos que precisam de uma definição mais completa no código.

Os modelos centrais são:

- neurônio Leaky Integrate-and-Fire (LIF);
- sinapse com peso e atraso;
- plasticidade dependente do tempo de disparo (STDP);
- plasticidade modulada por recompensa (R-STDP);
- regulação homeostática;
- métricas de atividade e esparsidade.

## 2. Modelo neuronal LIF

O modelo LIF representa a integração de corrente na membrana de um neurônio. O potencial diminui ao longo do tempo por causa do vazamento.

A equação diferencial usada no projeto é:

\[
\tau_m \frac{dV(t)}{dt}
=
-(V(t)-V_{rest}) + R I(t)
\]

Onde:

| Símbolo | Definição |
|---|---|
| \(V(t)\) | Potencial da membrana no tempo \(t\). |
| \(V_{rest}\) | Potencial de repouso. |
| \(\tau_m\) | Constante de tempo da membrana. |
| \(R\) | Resistência da membrana. |
| \(I(t)\) | Corrente total recebida pelas sinapses. |

### 2.1 Regra de disparo

O neurônio dispara quando o potencial atinge ou ultrapassa um limiar:

\[
V(t) \geq V_{th}
\]

O evento de disparo é representado por um pulso (*spike*). Após o disparo, o modelo reinicia o potencial para o estado de repouso ou para um valor de reinicialização definido na implementação.

O README descreve o reinício após o disparo. O valor exato do limiar, o período refratário e a regra de reinicialização devem ser confirmados no código e na configuração.

### 2.2 Forma discreta

Uma implementação por passos de tempo precisa discretizar a equação diferencial. Uma forma explícita de Euler, quando adotada, é:

\[
V_{t+\Delta t}
=
V_t +
\frac{\Delta t}{\tau_m}
\left[-(V_t-V_{rest}) + R I_t\right]
\]

Esta expressão é uma forma numérica possível para o modelo contínuo. O material fornecido não confirma que o código usa exatamente este integrador.

## 3. Corrente sináptica

Cada sinapse possui as propriedades descritas no projeto:

- peso \(w_{ij}\);
- atraso \(d_{ij}\);
- tipo excitatório ou inibitório;
- estado de plasticidade;
- histórico temporal de disparos.

Uma representação genérica da entrada de um neurônio \(j\) é:

\[
I_j(t) =
\sum_i w_{ij}\,s_i(t-d_{ij})
\]

Onde:

- \(i\) identifica o neurônio pré-sináptico;
- \(j\) identifica o neurônio pós-sináptico;
- \(s_i\) representa o sinal de disparo;
- \(d_{ij}\) representa o atraso sináptico.

Esta equação descreve uma forma geral de somar entradas. A implementação pode usar outra representação de corrente ou de eventos. Confirme a convenção de índices e sinais no código.

## 4. Plasticidade dependente do tempo de disparo (STDP)

A STDP altera o peso de uma sinapse com base na diferença entre os tempos de disparo pré-sináptico e pós-sináptico.

O projeto define:

\[
\Delta t = t_{pos} - t_{pre}
\]

A regra exponencial é:

\[
\Delta w =
\begin{cases}
A_+ e^{-\Delta t/\tau_+}, & \Delta t > 0\\
-A_- e^{\Delta t/\tau_-}, & \Delta t < 0
\end{cases}
\]

Onde:

| Símbolo | Definição |
|---|---|
| \(\Delta w\) | Alteração do peso sináptico. |
| \(A_+\) | Amplitude do fortalecimento. |
| \(A_-\) | Amplitude do enfraquecimento. |
| \(\tau_+\) | Constante temporal para o fortalecimento. |
| \(\tau_-\) | Constante temporal para o enfraquecimento. |
| \(\Delta t\) | Diferença entre os tempos dos disparos. |

### 4.1 Interpretação

- Se o neurônio pré-sináptico dispara antes do pós-sináptico, a regra aumenta o peso.
- Se o neurônio pós-sináptico dispara antes do pré-sináptico, a regra reduz o peso.

O caso \(\Delta t = 0\) não está definido na expressão original. A implementação deve definir esse comportamento.

### 4.2 Limites do peso

Para evitar valores sem limite, uma implementação pode restringir os pesos:

\[
w_{\min} \leq w_{ij} \leq w_{\max}
\]

O README não informa os limites usados. Verifique a implementação antes de registrar valores.

## 5. STDP modulada por recompensa (R-STDP)

O projeto descreve a modulação da plasticidade por um sinal de recompensa. A ideia é usar o resultado de uma ação para modular as alterações sinápticas.

Uma representação conceitual é:

\[
\Delta w_{ij} = M(t)\,e_{ij}(t)
\]

Onde:

- \(M(t)\) é o sinal modulador de recompensa;
- \(e_{ij}(t)\) é um traço de elegibilidade associado à atividade da sinapse.

Esta expressão é uma representação geral de R-STDP. O README e os logs fornecidos não especificam a equação exata do traço de elegibilidade nem a dinâmica do sinal modulador.

A implementação deve documentar:

1. como calcula a recompensa;
2. como atualiza o traço de elegibilidade;
3. quanto tempo o traço permanece ativo;
4. como aplica a recompensa ao peso;
5. quais limites se aplicam ao peso.

## 6. Plasticidade homeostática

A plasticidade homeostática busca manter a atividade neural em uma faixa estável. O README informa que o sistema regula o limiar de disparo.

Uma representação conceitual de uma atualização de limiar é:

\[
\theta_i(t+\Delta t)
=
\theta_i(t) + \eta_h\left(r_i(t)-r_{target}\right)
\]

Onde:

- \(\theta_i\) é o limiar do neurônio \(i\);
- \(\eta_h\) é a taxa de adaptação homeostática;
- \(r_i\) é a taxa de disparo observada;
- \(r_{target}\) é a taxa de disparo desejada.

Esta é uma formulação ilustrativa de controle homeostático. O material fornecido não confirma que o código usa esta equação. Consulte a implementação de `homeostasis.py` para registrar a regra real.

## 7. Codificação de estímulos

O projeto descreve codificadores para sensores, visão e áudio. Os codificadores convertem valores contínuos ou sinais de entrada em pulsos.

Duas famílias comuns de codificação são:

### 7.1 Codificação por taxa

A informação é representada pela frequência dos disparos em uma janela de tempo.

\[
r = \frac{N_{spikes}}{T}
\]

Onde:

- \(r\) é a taxa de disparo;
- \(N_{spikes}\) é o número de pulsos;
- \(T\) é a duração da janela.

### 7.2 Codificação temporal

A informação é representada pelo instante em que cada pulso ocorre.

O README menciona codificação por taxa e temporal. Os logs mostram saídas binárias dos codificadores. Eles não especificam a regra completa de codificação usada por cada modalidade.

## 8. Conectividade esparsa

O projeto usa matrizes esparsas para representar os pesos sinápticos.

A densidade de conexões pode ser descrita por:

\[
\rho = \frac{N_{conexões}}{N_{possíveis}}
\]

Onde:

- \(\rho\) é a densidade;
- \(N_{conexões}\) é o número de conexões armazenadas;
- \(N_{possíveis}\) é o número de conexões possíveis na topologia considerada.

A proporção de posições sem conexão é:

\[
1-\rho
\]

Os testes registraram densidades de 15%, 5% e 1% para as escalas de 10 mil, 100 mil e 1 milhão de neurônios, respectivamente.

A topologia e o tratamento de autoconexões devem ser documentados para definir \(N_{possíveis}\) sem ambiguidade.

## 9. Métricas registradas

### 9.1 Taxa de disparo

\[
r = \frac{N_{spikes}}{N_{neurônios}\,T}
\]

Esta forma calcula a taxa média por neurônio e por unidade de tempo. O log de 10 mil neurônios informa a média de pulsos por passo, mas não informa a duração física de cada passo.

### 9.2 Taxa de sucesso

Uma forma comum de calcular a taxa de sucesso é:

\[
P_{sucesso} =
\frac{N_{tarefas\ concluídas}}{N_{tarefas\ tentadas}}\times 100
\]

O log de monitoramento registra 80%. O material fornecido não informa os numeradores e denominadores usados. Portanto, não é possível reconstruir o cálculo.

### 9.3 Operações sinápticas

O projeto usa a métrica SOPs (*Synaptic Operations*). Os logs registram:

- 5.781.512 SOPs no teste de 100 mil neurônios;
- 69.163.386 SOPs no teste de 1 milhão de neurônios.

O método de contagem não aparece nos materiais fornecidos. Documente quais operações entram nessa métrica.

### 9.4 Estimativa de energia

Os logs registram estimativas de energia:

| Escala | Energia estimada |
|---:|---:|
| Teste de monitoramento | 0,00000047 J |
| 100.000 neurônios | 0,00003188 J |
| 1.000.000 de neurônios | 0,00023616 J |

O método de estimativa não está documentado nos materiais fornecidos. Não trate esses valores como medições físicas até validar a fórmula ou usar um instrumento de medição.

## 10. Parâmetros que precisam de registro

Mantenha uma tabela de parâmetros por experimento:

| Parâmetro | Símbolo | Estado da documentação |
|---|---|---|
| Potencial de repouso | \(V_{rest}\) | Nomeado no README; valor não fornecido. |
| Limiar de disparo | \(V_{th}\) | Necessário para a regra de disparo; valor não fornecido. |
| Constante de membrana | \(\tau_m\) | Nomeada no README; valor não fornecido. |
| Resistência da membrana | \(R\) | Nomeada no README; valor não fornecido. |
| Passo de tempo | \(\Delta t\) | Usado conceitualmente; valor não fornecido. |
| Amplitude STDP positiva | \(A_+\) | Nomeada no README; valor não fornecido. |
| Amplitude STDP negativa | \(A_-\) | Nomeada no README; valor não fornecido. |
| Constante STDP positiva | \(\tau_+\) | Nomeada no README; valor não fornecido. |
| Constante STDP negativa | \(\tau_-\) | Nomeada no README; valor não fornecido. |
| Taxa homeostática | \(\eta_h\) | Não informada. |
| Recompensa | \(M(t)\) | Regra exata não informada. |

## 11. Referências de implementação

O README do projeto indica estes arquivos como locais esperados dos modelos:

```text
core/neuron.py
core/synapse.py
core/spike_engine.py
core/simulation.py
learning/stdp.py
learning/homeostasis.py
learning/reward_learning.py
perception/sensor_encoder.py
perception/vision.py
perception/audio.py
monitoring/performance.py
config/brain_config.yaml
```


