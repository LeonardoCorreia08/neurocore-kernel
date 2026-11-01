# Registros de experimentos do NEUROCORE

## 1. Escopo

Este documento consolida os registros de terminal fornecidos para o NEUROCORE.

Os resultados abaixo são transcrições resumidas dos logs. Este documento não acrescenta resultados que não aparecem nesses registros.

O log não informa a data de cada execução, o modelo de CPU, a GPU, a memória disponível, a versão do Python ou a versão do PyTorch. Registre esses dados nas próximas execuções.

## 2. Resumo dos resultados

| Teste | Resultado | Estado observado |
|---|---|---|
| `test_05_scale_10k` | 10 mil neurônios; 50 passos; 17,4204 s | Concluído |
| `test_06_neurocore_03_agent` | 1 meta em 10 episódios | Concluído após uma falha anterior |
| `neurocore_04_test` | 5 episódios; checkpoint salvo | Concluído |
| `test_08_cortical_modules` | 181 metas acumuladas em 50 episódios | Concluído |
| `test_09_memory_modules` | checkpoint, associação e poda | Testes concluídos; associação requer revisão |
| `test_10_perception_modules` | codificadores de sensor, visão e áudio | Concluído |
| `test_11_monitoring` | frequência, desvio, energia estimada e taxa de acerto | Concluído após correções |
| `test_12_scale_100k` | 100 mil neurônios; ciclo de 174,40 ms | Concluído |
| `test_13_scale_1m` | 1 milhão de neurônios; ciclo de 2.394,65 ms | Concluído |
| `test_14_generate_plots` | três gráficos salvos | Concluído |

## 3. Teste de escala: 10 mil neurônios

**Comando:**

```bash
python -m experiments.test_05_scale_10k
```

**Configuração registrada:**

- População: 10.000 neurônios.
- Densidade esparsa: 15%.
- Simulação: 50 passos.

**Resultados:**

- Construção da rede: 5,45 s.
- Tempo total de simulação: 17,4204 s.
- Média de pulsos por passo: 4.577,72.

Valores registrados em passos selecionados:

| Passo | Spikes |
|---:|---:|
| 10 | 4.656 |
| 20 | 4.560 |
| 30 | 4.658 |
| 40 | 4.532 |
| 50 | 4.642 |

**Estado:** o teste terminou com sucesso.

**Aviso:** o PyTorch informou que as verificações de invariantes dos tensores esparsos estavam desativadas. O aviso cita risco de erros de memória se um tensor esparso violar os invariantes.

## 4. Agente NeuroCore 0.3: 10 mil neurônios e R-STDP

**Comando:**

```bash
python -m experiments.test_06_neurocore_03_agent
```

### 4.1 Primeira execução

O agente iniciou com 10 mil neurônios e duas matrizes esparsas:

- W1: sensorial para oculta;
- W2: oculta para motora.

O teste parou no episódio 3. A atualização de pesos gerou um erro de dimensão:

```text
The size of tensor a (5586) must match the size of tensor b (4)
at non-singleton dimension 1
```

O erro ocorreu durante a soma entre os valores dos pesos esparsos e os valores de atualização.

### 4.2 Segunda execução

| Episódio | Passos | Metas atingidas acumuladas |
|---:|---:|---:|
| 1 | 100 | 0 |
| 2 | 100 | 0 |
| 3 | 100 | 0 |
| 4 | 100 | 0 |
| 5 | 32 | 1 |
| 6 | 100 | 1 |
| 7 | 100 | 1 |
| 8 | 100 | 1 |
| 9 | 100 | 1 |
| 10 | 100 | 1 |

**Estado:** a segunda execução terminou. O log não descreve a alteração que resolveu o erro da primeira execução.

**Interpretação permitida:** o agente atingiu uma meta durante a execução registrada. O resultado não demonstra convergência ou generalização.

## 5. NeuroCore 0.4: atenção, obstáculos e persistência

**Comando:**

```bash
python -m experiments.neurocore_04_test
```

**Resultados:**

| Episódio | Passos |
|---:|---:|
| 1 | 80 |
| 2 | 80 |
| 3 | 16 |
| 4 | 80 |
| 5 | 80 |

O teste salvou um checkpoint sináptico em:

```text
config/synapse_w1_checkpoint.pt
```

**Estado:** o teste terminou com sucesso, conforme o log.

**Limite:** o registro não apresenta uma métrica isolada para atenção cognitiva.

## 6. Teste de módulos corticais

**Comando:**

```bash
python -m experiments.test_08_cortical_modules
```

O teste executou 50 episódios, com até 200 passos por episódio.

O log registrou os seguintes valores acumulados:

| Episódio | Metas acumuladas |
|---:|---:|
| 1–6 | 0 |
| 7–12 | 2 |
| 13–24 | 4 |
| 25–42 | 173 |
| 43–50 | 181 |

**Estado:** o teste terminou com sucesso.

**Limite:** o log informa metas acumuladas. Ele não informa a taxa por episódio, a quantidade de tentativas, a distribuição dos mapas ou uma comparação com uma política de referência.

## 7. Testes de memória

**Comando:**

```bash
python -m experiments.test_09_memory_modules
```

O primeiro registro dessa execução mostra 50 episódios e 19 metas acumuladas. Em seguida, o log mostra uma execução específica dos subsistemas de memória.

### 7.1 Persistência de longo prazo

O sistema salvou e carregou:

```text
memory/storage/neurocore_snapshot_test.pt
```

O estado carregado incluiu:

```text
weights_sample: [0.5, -0.2, 0.8]
epoch: 10
```

**Resultado:** o log confirma a gravação e a leitura do checkpoint de teste.

### 7.2 Memória associativa

O teste registrou:

```text
[Associative Test] Padrao recuperado a partir de A:
tensor([0., 0., 0., 0.])
```

**Resultado:** houve uma saída do módulo. O vetor zerado não confirma a recuperação correta do padrão esperado.

**Ação necessária:** definir o padrão de entrada, o padrão esperado e um critério de igualdade ou similaridade. Depois, repetir o teste.

### 7.3 Consolidação e poda

O teste registrou:

- 330 sinapses fracas removidas;
- 1.467 sinapses ativas restantes.

**Estado:** a poda foi executada com sucesso, conforme o log.

## 8. Teste de percepção

**Comando:**

```bash
python -m experiments.test_10_perception_modules
```

Saídas registradas:

| Componente | Saída |
|---|---|
| Sensor Encoder | `[0., 1., 0., 0.]` |
| Vision Encoder | vetor de dimensão 8 |
| Audio Encoder | `[0., 1., 1., 1.]` |
| Neural Integration | pulsos propagados em camada LIF |

O potencial de membrana resultante foi:

```text
[0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5]
```

**Estado:** o teste terminou com sucesso.

**Limite:** o teste demonstra a execução dos codificadores com os valores usados. Não demonstra precisão em bases reais de áudio ou imagem.

## 9. Teste de monitoramento

**Comando:**

```bash
python -m experiments.test_11_monitoring
```

### 9.1 Primeira tentativa

O teste parou por um erro de sintaxe em `monitoring/spike_visualizer.py`.

### 9.2 Segunda tentativa

O teste parou porque não foi possível importar `SpikeVisualizer`.

### 9.3 Terceira tentativa

O teste terminou e registrou:

| Métrica | Valor |
|---|---:|
| Histórico de spikes | 50 passos |
| Desvio homeostático | 4,62 |
| Custo energético estimado | 0,00000047 J |
| Taxa de acerto | 80% |

**Estado:** concluído após correções.

**Limite:** o log não descreve a fórmula usada para estimar energia nem o protocolo usado para calcular a taxa de acerto.

## 10. Teste de escala: 100 mil neurônios

**Comando:**

```bash
python -m experiments.test_12_scale_100k
```

**Configuração registrada:**

- População: 100.000 neurônios.
- Densidade: 5%.

**Resultados:**

| Métrica | Valor |
|---|---:|
| Tempo de alocação e coalescência | 1,92 s |
| Tempo do ciclo de teste | 174,40 ms |
| Spikes ativos | 26.094 |
| Operações sinápticas (SOPs) | 5.781.512 |
| Custo energético estimado | 0,00003188 J |

**Estado:** o teste terminou com sucesso.

## 11. Teste de escala: 1 milhão de neurônios

**Comando:**

```bash
python -m experiments.test_13_scale_1m
```

**Configuração registrada:**

- População: 1.000.000 neurônios.
- Densidade: 1%.

**Resultados:**

| Métrica | Valor |
|---|---:|
| Tempo de alocação e coalescência | 29,25 s |
| Tempo do ciclo de teste | 2.394,65 ms |
| Spikes ativos | 166.992 |
| Operações sinápticas (SOPs) | 69.163.386 |
| Custo energético estimado | 0,00023616 J |

**Estado:** o teste terminou com sucesso.

**Interpretação:** o log demonstra que a rede foi alocada e que um ciclo de teste foi executado nessa escala. Não demonstra aprendizagem, retenção ou generalização com um milhão de neurônios.

## 12. Geração de gráficos

**Comando:**

```bash
python -m experiments.test_14_generate_plots
```

O teste registrou a criação de:

```text
experiments/plots/neurocore_learning_curve.png
experiments/plots/neurocore_raster_plot.png
experiments/plots/neurocore_energy_profile.png
```

**Estado:** os três arquivos foram gerados, conforme o log.

## 13. Problemas técnicos observados

| Problema | Evidência | Ação recomendada |
|---|---|---|
| Invariantes esparsos desativados | Aviso do PyTorch em diferentes testes | Validar índices e valores. Avaliar a ativação das verificações durante os testes. |
| Dimensões incompatíveis na R-STDP | Erro entre 5.586 e 4 elementos | Verificar alinhamento entre índices esparsos e deltas de peso. |
| Erro de sintaxe no visualizador | `SyntaxError` | Manter teste de importação e lint. |
| Falha de importação | `SpikeVisualizer` não encontrado | Conferir nome da classe e módulo. |
| Recuperação associativa zerada | Vetor `[0, 0, 0, 0]` | Criar teste com padrão conhecido e resultado esperado. |

## 14. Protocolo recomendado para os próximos experimentos

Registre estes dados em cada execução:

1. Identificador e versão do experimento.
2. Data e hora.
3. Commit do código.
4. Sistema operacional.
5. CPU, GPU e memória.
6. Versões de Python, PyTorch e CUDA.
7. Semente aleatória.
8. Número de neurônios e sinapses.
9. Densidade e topologia da rede.
10. Número de passos e episódios.
11. Métrica, unidade e método de cálculo.
12. Resultado bruto e estado final.

Repita cada experimento com mais de uma semente. Informe média e desvio-padrão. Use a mesma carga de trabalho quando comparar escalas.

## 15. Nota sobre energia

Os logs apresentam valores em joules. O material fornecido não mostra o método de medição ou a fórmula de estimativa.

Até que o método seja documentado e validado, descreva esses números como **estimativas produzidas pelo módulo de avaliação**. Não os apresente como medições físicas de consumo da CPU ou GPU.
