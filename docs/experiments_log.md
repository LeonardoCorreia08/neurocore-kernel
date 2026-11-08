# NEUROCORE — Registro de Experimentos

Este documento consolida os resultados impressos nos registros de execução fornecidos para os experimentos do NEUROCORE. Os valores são transcritos dos logs disponíveis; não representam medições independentes nem garantem que os testes sejam reproduzíveis em outro ambiente.

## 1. Resumo dos experimentos

| Experimento | Objetivo declarado | Resultado registrado |
|---|---|---|
| `test_01_lif_base` | Testar o agente neural base (NeuroCore 0.1). | 50 episódios de 200 passos; 6 metas atingidas acumuladas. |
| `test_02_stdp` | Testar o aprendizado STDP. | 50 episódios de 200 passos; 6 metas atingidas. |
| `test_03_rstdp` | Testar o aprendizado R-STDP modulado por recompensa. | 50 episódios; 5 metas atingidas. Alguns episódios terminaram antes de 200 passos. |
| `test_04_neurocore_02` | Testar NeuroCore 0.2 com Memory Engine e Raster Plot. | 30 episódios; 1 meta atingida; raster plot salvo. |
| `test_05_scale_10k` | Avaliar uma rede esparsa com 10.000 neurônios. | Construção da rede em 5,16 s; simulação de 50 passos em 17,5334 s. |
| `test_06_neurocore_03_agent` | Testar agente NeuroCore 0.3 com 10.000 neurônios e R-STDP. | 10 episódios; 1 meta atingida. |
| `test_07_neurocore` | Testar NeuroCore 0.4 com atenção, obstáculos e persistência. | 5 episódios de 80 passos; checkpoint sináptico salvo. |
| `test_08_cortical_modules` | Integrar subsistemas corticais. | Ciclo de 20 passos concluído; nenhuma recompensa positiva ou estado final foi registrado nos passos exibidos. |
| `test_09_memory_modules` | Testar persistência, recuperação associativa e poda. | Snapshot salvo e carregado; padrão `[0., 1., 0., 1.]` recuperado; 380 sinapses podadas e 1.416 ativas restantes. |
| `test_10_perception_modules` | Testar codificadores sensorial, visual e auditivo. | Os codificadores produziram os vetores de pulsos registrados no log. |
| `test_11_monitoring` | Testar avaliação e telemetria. | 50 passos registrados; desvio homeostático 5,82; custo energético estimado de 0,00000047 J; taxa de acerto de 80%. |
| `test_12_scale_100k` | Testar alocação e ciclo com 100.000 neurônios. | Alocação em 2,02 s; ciclo em 197,90 ms; 24.341 spikes ativos. |
| `test_13_scale_1m` | Testar alocação e ciclo com 1.000.000 de neurônios. | Alocação em 29,87 s; ciclo em 2.376,34 ms; 174.005 spikes ativos. |
| `test_14_generate_plots` | Gerar gráficos dos experimentos. | Três arquivos de gráfico foram salvos. |
| `test_15_generalization` | Testar adaptação no Ambiente A e transferência para o Ambiente B. | Eficiência de 70% no Ambiente A e taxa de sucesso registrada de 100% no Ambiente B. |
| `test_16_dynamic_1m` | Testar simulação dinâmica com 1.000.000 de neurônios. | O cabeçalho anuncia 1.000 passos, mas o log registra 50 passos; tempo total de 50,28 s. |
| `test_17_homeostasis_recovery` | Validar a recuperação homeostática e limites de atividade. | Passou. Ajuste de limiar estabilizou superatividade. |
| `test_18_learning_retention` | Validar consolidação na memória de longo prazo (LTM). | Passou. Memória apagada e restaurada com exatidão. |
| `test_19_synapse_consistency` | Testar limites e poda da matriz esparsa. | Passou. 47.559 sinapses ativas podadas para 23.780. |
| `test_20_homeostasis` | Validação de estabilidade a longo prazo. | Passou. Limiares adaptados para manter a estabilidade. |
| `test_21_full_integration` | Integração total (Itens 6 a 13). | Passou. Recompensa processada e reset bem-sucedido. |
| `ablation_study` | Estudo de ablação (Precisão Direcional). | Rede adaptativa (56,44%) superou rede estática (20,0%). |
| `attention_decision_validation` | Atenção seletiva vs estímulos distratores. | Concluído com aviso sobre diferenciação de padrões. |
| `continual_learning_retention` | Esquecimento catastrófico. | Retenção subiu para 40% com o uso de replay episódico. |
| `continuous_learning_generalization`| Generalização contínua. | Concluído, mas a taxa impressa (370.000.000%) é uma anomalia. |
| `evaluate_saved_checkpoint` | Avaliar checkpoint salvo. | **Falha de execução** (AttributeError). |
| `latency_profile` | Benchmark de latência por passo. | 2,93 ms/tick e Throughput de 341,76 passos/segundo. |
| `memory_noise_validation` | Recuperação temporal sob 30% de ruído. | Passou. Similaridade funcional de 94,77%. |
| `memory_profile` | Profiling de RAM/VRAM por escala. | RAM máxima 2,84MB. VRAM 0MB (rodou em CPU). |
| `memory_test` | Treino, Snapshot, Wipe e Restore. | Passou. Comportamento restaurado idêntico. |
| `million_neuron_scale_test` | Benchmark unificado (10k, 100k, 1M). | 1M neurônios, latência forward W*x de 8,37 ms. |
| `neural_dynamics_analysis` | Dinâmica (estabilidade & frequência). | Camada Oculta estabilizou em ~62Hz. |
| `neuron_test` | Teste de unidade de neurônio. | **Falha de sintaxe** (Erro de codificação utf-8). |
| `plasticity_validation` | Validação de STDP e R-STDP. | Passou. Plasticidade temporal e dopaminérgica ativas. |
| `reinforcement_learning_assessment`| Avaliação RL. | **Falha de execução** (TypeError em VirtualWorld2D). |
| `scalability_stress_test` | Estresse alterando densidade esparsa. | Rede Deep SNN suportou a propagação a 12,64 ms/tick. |
| `surrogate_gradient_validation` | Validação de BPTT. | Acurácia de spikes 56,2% (C.I. 95%). |
| `test_directional_sensors` | Sensores direcionais virtuais. | **Falha de execução** (TypeError). |
| `test_evaluate_saved_checkpoint` | Testar avaliação de checkpoint salvo. | **Falha de execução** (AttributeError). |
| `test_td_critic` | Teste do crítico TD(0). | Passou. |
| `test_training_resume` | Retomar treinamento. | **Falha de execução** (TypeError). |

## 2. Experimentos de aprendizado

### 2.1 NeuroCore 0.1 — modelo base

**Comando**

```powershell
python -m experiments.test_01_lif_base
```

**Registros**

- Episódios executados: 50.
- Passos informados por episódio: 200.
- Metas atingidas acumuladas ao final: 6.
- O log informa que o experimento foi concluído com sucesso.

O resultado corresponde à contagem de metas impressa pelo teste. O log, isoladamente, não define a dificuldade das metas nem demonstra generalização para outros ambientes.

### 2.2 NeuroCore 0.1 — STDP

**Comando**

```powershell
python -m experiments.test_02_stdp
```

**Registros**

- Episódios executados: 50.
- Passos informados por episódio: 200.
- Metas atingidas ao final: 6.
- O log informa que o experimento com STDP foi concluído com sucesso.

Não há, nesse registro, uma comparação controlada que isole o efeito do STDP em relação ao experimento base.

### 2.3 NeuroCore 0.1 — R-STDP

**Comando**

```powershell
python -m experiments.test_03_rstdp
```

**Registros**

- Episódios executados: 50.
- Metas atingidas ao final: 5.
- Alguns episódios terminaram antes de 200 passos; o log apresenta, por exemplo, episódios com 8, 3, 10, 2, 24, 17, 5, 1, 13, 7 e 12 passos.
- O log informa que o experimento foi concluído com sucesso.

A contagem de metas não deve ser comparada diretamente com os outros testes sem confirmar se ambiente, sementes, condições iniciais e critérios de encerramento eram equivalentes.

### 2.4 NeuroCore 0.2 — memória e raster plot

**Comando**

```powershell
python -m experiments.test_04_neurocore_02
```

**Registros**

- Episódios executados: 30.
- Passos configurados: 200 por episódio, embora alguns episódios tenham terminado antes.
- Metas atingidas ao final: 1.
- Raster plot salvo em `docs/spike_raster.png`.

O arquivo de imagem é indicado pelo log; sua presença e conteúdo devem ser verificados no repositório.

### 2.5 NeuroCore 0.3 — agente com 10.000 neurônios

**Comando**

```powershell
python -m experiments.test_06_neurocore_03_agent
```

**Registros**

- Arquitetura descrita no log: 10.000 neurônios, R-STDP e ambiente 2D.
- Episódios executados: 10.
- Passos configurados: 100 por episódio.
- Metas atingidas: 1.
- O log exibe um aviso do PyTorch sobre verificações de invariantes de tensores esparsos desativadas.

O aviso é relevante para a confiabilidade das operações com tensores esparsos e deve ser investigado. A conclusão de execução do teste não elimina esse aviso.

### 2.6 NeuroCore 0.4 — atenção, obstáculos e persistência

**Comando**

```powershell
python -m experiments.test_07_neurocore
```

**Registros**

- Episódios concluídos: 5.
- Passos por episódio: 80.
- Checkpoint salvo em `config/synapse_w1_checkpoint.pt`.
- O log informa conclusão e validação dentro da estrutura modular.

O salvamento do checkpoint demonstra que o arquivo foi gravado segundo o log; não demonstra, sozinho, que todos os estados podem ser restaurados e continuar o aprendizado corretamente.

## 3. Testes de subsistemas

### 3.1 Subsistemas corticais

**Comando**

```powershell
python -m experiments.test_08_cortical_modules
```

**Registros**

- Ciclo de integração executado: 20 passos.
- Ações registradas: principalmente ação `0`, com ação `2` em alguns passos.
- Recompensa exibida nos 20 passos: `0.0`.
- Campo `Feito`: `False` em todos os passos exibidos.
- O log informa conclusão do teste integrado.

A execução do ciclo não equivale a demonstrar que o agente atingiu uma meta nesse teste.

### 3.2 Subsistemas de memória

**Comando**

```powershell
python -m experiments.test_09_memory_modules
```

**Registros**

- Snapshot de memória de longo prazo salvo em `memory/storage/neurocore_snapshot_test.pt`.
- Snapshot carregado a partir do mesmo caminho.
- Estado carregado: amostra de pesos `[0.5000, -0.2000, 0.8000]` e `epoch: 10`.
- Padrão associativo recuperado: `[0., 1., 0., 1.]`.
- Sinapses fracas podadas: 380.
- Sinapses ativas restantes: 1.416.
- O log informa que os testes dos subsistemas de memória foram concluídos.

Esses resultados documentam a saída desse teste específico. A robustez da memória exige repetir o teste com padrões, ruído, tamanhos e condições variados.

### 3.3 Módulos de percepção

**Comando**

```powershell
python -m experiments.test_10_perception_modules
```

**Saídas registradas**

- Codificador de sensor: `[0., 1., 1., 0.]`.
- Codificador visual, dimensão 8: `[1., 1., 1., 0., 1., 0., 1., 0.]`.
- Codificador de áudio: `[0., 1., 0., 0.]`.
- Integração neural: `[1., 1., 1., 0., 1., 0., 1., 0.]`.
- Potencial de membrana resultante: `[0.5000, 0.5000, 0.5000, 0.0000, 0.5000, 0.0000, 0.5000, 0.0000]`.

O teste demonstra a produção dos vetores indicados no log. Não documenta, por si só, desempenho em imagens ou áudio do mundo real.

### 3.4 Avaliação e telemetria

**Comando**

```powershell
python -m experiments.test_11_monitoring
```

**Registros**

- Histórico do visualizador: 50 passos.
- Desvio homeostático: 5,82.
- Custo energético estimado: `0.00000047 Joules`.
- Taxa de acerto: 80%.
- O log informa conclusão do teste.

O custo energético é explicitamente descrito como **estimado**. O log não fornece uma medição física de consumo de energia nem detalhes suficientes para validar a estimativa.

## 4. Testes de escalabilidade

### 4.1 Rede de 10.000 neurônios

**Comando**

```powershell
python -m experiments.test_05_scale_10k
```

**Configuração e resultados registrados**

| Métrica | Valor |
|---|---:|
| Neurônios | 10.000 |
| Densidade esparsa declarada | 15,0% |
| Construção da rede esparsa | 5,16 s |
| Passos simulados | 50 |
| Tempo total de simulação | 17,5334 s |
| Média de spikes por passo | 4.585,72 |

O log contém um aviso do PyTorch indicando que as verificações de invariantes de tensores esparsos estão desativadas. O registro informa conclusão do teste.

### 4.2 Rede de 100.000 neurônios

**Comando**

```powershell
python -m experiments.test_12_scale_100k
```

**Resultados registrados**

| Métrica | Valor |
|---|---:|
| Neurônios | 100.000 |
| Densidade esparsa declarada | 5,0% |
| Alocação e coalescência | 2,02 s |
| Tempo do ciclo de teste | 197,90 ms |
| Spikes ativos no ciclo | 24.341 |
| Operações sinápticas (SOPs) | 5.781.363 |
| Custo energético estimado | 0,00003012 J |

O log informa que a alocação e o ciclo de teste foram concluídos. O valor energético é uma estimativa impressa pelo módulo, não uma medição física documentada.

### 4.3 Rede de 1.000.000 de neurônios

**Comando**

```powershell
python -m experiments.test_13_scale_1m
```

**Resultados registrados**

| Métrica | Valor |
|---|---:|
| Neurônios | 1.000.000 |
| Densidade esparsa declarada | 1,0% |
| Alocação e coalescência | 29,87 s |
| Tempo do ciclo de teste | 2.376,34 ms |
| Spikes ativos no ciclo cortical | 174.005 |
| Operações sinápticas (SOPs) | 69.164.589 |
| Custo energético estimado | 0,00024317 J |

O log informa que a rede foi alocada e que um ciclo foi executado. Isso demonstra a execução do teste estrutural nas condições registradas; não comprova aprendizagem útil, desempenho cognitivo geral ou eficiência energética medida em hardware nessa escala.

### 4.4 Comparabilidade dos testes de escala

As densidades esparsas declaradas diferem entre os testes: 15% para 10 mil neurônios, 5% para 100 mil e 1% para 1 milhão. Portanto, os resultados não constituem uma comparação controlada em que apenas o número de neurônios muda.

Os logs fornecidos não registram todos os dados necessários para uma reprodução completa, como especificação integral do hardware, versão exata do PyTorch, carga concorrente, sementes aleatórias e detalhes completos da configuração de cada execução.

## 5. Geração de gráficos

**Comando**

```powershell
python -m experiments.test_14_generate_plots
```

**Arquivos informados pelo log**

- `experiments/plots/neurocore_learning_curve.png`
- `experiments/plots/neurocore_raster_plot.png`
- `experiments/plots/neurocore_energy_profile.png`

O log informa que os três gráficos foram gerados. A interpretação dos gráficos requer examinar os arquivos e verificar quais dados foram usados para produzi-los.

## 6. Generalização e transferência de aprendizado

**Comando**

```powershell
python -m experiments.test_15_generalization
```

**Registros**

- Adaptação sináptica executada no Ambiente A.
- Eficiência registrada no Ambiente A: 70,0%.
- Avaliação de transferência para o Ambiente B, descrito no log como inédito e gerado proceduralmente.
- Taxa de sucesso registrada no Ambiente B: 100,0%.

Esses valores são os resultados impressos pelo teste. O log não detalha o número de tentativas, a definição de eficiência e sucesso, a diversidade dos ambientes, a semente aleatória nem o protocolo de repetição. Assim, o resultado não deve ser generalizado para outros ambientes sem validação adicional.

## 7. Teste dinâmico com 1.000.000 de neurônios

**Comando**

```powershell
python -m experiments.test_16_dynamic_1m
```

**Registros**

- O cabeçalho anuncia um teste de 1.000 passos.
- O loop também é descrito como tendo 1.000 passos.
- O relatório final registra **50 passos executados**.
- Tempo total de simulação: 50,28 s.
- Média de spikes ativos por passo informada: 228.283.
- O campo de consumo energético real medido registra `0.000000 Joules`.

**Inconsistências e limites**

1. O número anunciado de passos (1.000) não corresponde ao total informado no relatório (50). O registro disponível não comprova a execução de 1.000 passos.
2. O consumo energético aparece como `0.000000 Joules`. Sem método de medição, resolução do instrumento ou explicação do procedimento, esse valor não deve ser interpretado como prova de consumo físico nulo.
3. O log apresenta contagens de spikes em passos selecionados, mas não fornece todos os dados brutos da simulação.

É necessário inspecionar a implementação de `test_16_dynamic_1m` e repetir o teste para determinar por que somente 50 passos foram registrados.

## 8. Relatórios de Erros e Falhas Técnicas (Crashes)

Durante a execução da bateria de testes adicionais, cinco experimentos específicos apresentaram interrupção forçada (`Traceback`). Os erros foram consolidados abaixo para orientar a correção do código-fonte.

**1. `evaluate_saved_checkpoint`** e **`test_evaluate_saved_checkpoint`**
```text
AttributeError: 'ContinualLearningAgent' object has no attribute 'restore_memory'
AttributeError: 'ContinualLearningAgent' object has no attribute 'save_checkpoint'
```
* **Diagnóstico:** A classe `ContinualLearningAgent` não implementa ou teve seus métodos de checkpointing (`restore_memory` / `save_checkpoint`) renomeados ou movidos para outro módulo (como o `MemoryEngine`).

**2. `test_directional_sensors`** e **`reinforcement_learning_assessment`**
```text
TypeError: VirtualWorld2D.__init__() got an unexpected keyword argument 'sensor_mode'
```
* **Diagnóstico:** O construtor (`__init__`) da classe de ambiente `VirtualWorld2D` não suporta o parâmetro `sensor_mode`. A assinatura da classe mudou e os testes não foram atualizados.

**3. `test_training_resume`**
```text
TypeError: run_training_and_visualization() got an unexpected keyword argument 'seed'
```
* **Diagnóstico:** A função chamada no script de teste não suporta o argumento nomeado `seed` em sua definição atual.

**4. `neuron_test`**
```text
SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0xf3 in position 1: invalid continuation byte
```
* **Diagnóstico:** O arquivo `neuron_test.py` contém caracteres não reconhecidos (possivelmente acentuação) salvos em um formato diferente de UTF-8 logo na sua primeira linha (`'''Módulo...'''`).

## 9. Avisos técnicos observados

Os logs de redes e sinapses esparsas apresentam repetidamente o aviso do PyTorch:

> Sparse invariant checks are implicitly disabled.

O próprio aviso informa que operar sobre um tensor esparso que viole invariantes pode causar erros de memória, incluindo falha de segmentação, enquanto as verificações têm custo de desempenho. O aviso não prova que ocorreu corrupção de memória, mas indica um risco que deve ser investigado.

Ações recomendadas:

1. Verificar como os tensores esparsos são construídos em `core/sparse_network.py` e `core/synapse.py`.
2. Validar índices, dimensões, valores e invariantes dos tensores antes de operações críticas.
3. Criar testes específicos para construção, atualização, coalescência, salvamento e carregamento de pesos esparsos.
4. Registrar versões de Python, PyTorch, CUDA e bibliotecas relevantes em cada execução.
5. Repetir os testes com sementes controladas e múltiplas execuções.
6. **Priorizar as correções de código das assinaturas de classes e imports descritas na Seção 8.**

## 10. Conclusões permitidas pelos registros

Os logs fornecidos documentam:

- execução de testes dos modelos base, STDP e R-STDP;
- execução de módulos de memória, percepção, integração cortical e monitoramento;
- geração de arquivos de gráficos;
- alocação de redes com 10 mil, 100 mil e 1 milhão de neurônios;
- execução de ciclos de teste nas escalas de 100 mil e 1 milhão de neurônios;
- resultados de metas e taxas impressas por testes específicos;
- **a ocorrência de 6 falhas críticas na arquitetura de testes devido à dessincronização de interfaces (API).**

Os registros, isoladamente, **não comprovam** inteligência geral, capacidade cognitiva robusta em escala de um milhão de neurônios, generalização ampla, eficiência energética física ou reprodutibilidade independente. Essas afirmações exigem definições operacionais, protocolos controlados, dados brutos e validações adicionais.

## 11. Rastreabilidade dos comandos

Os comandos abaixo aparecem nos registros fornecidos:

```powershell
python -m experiments.test_01_lif_base
python -m experiments.test_02_stdp
python -m experiments.test_03_rstdp
python -m experiments.test_04_neurocore_02
python -m experiments.test_05_scale_10k
python -m experiments.test_06_neurocore_03_agent
python -m experiments.test_07_neurocore
python -m experiments.test_08_cortical_modules
python -m experiments.test_09_memory_modules
python -m experiments.test_10_perception_modules
python -m experiments.test_11_monitoring
python -m experiments.test_12_scale_100k
python -m experiments.test_13_scale_1m
python -m experiments.test_14_generate_plots
python -m experiments.test_15_generalization
python -m experiments.test_16_dynamic_1m
python -m experiments.test_17_homeostasis_recovery
python -m experiments.test_18_learning_retention
python -m experiments.test_19_synapse_consistency
python -m experiments.test_20_homeostasis
python -m experiments.test_21_full_integration
python -m experiments.ablation_study
python -m experiments.attention_decision_validation
python -m experiments.continual_learning_retention
python -m experiments.continuous_learning_generalization
python -m experiments.evaluate_saved_checkpoint
python -m experiments.latency_profile
python -m experiments.memory_noise_validation
python -m experiments.memory_profile
python -m experiments.memory_test
python -m experiments.million_neuron_scale_test
python -m experiments.neural_dynamics_analysis
python -m experiments.neuron_test
python -m experiments.plasticity_validation
python -m experiments.reinforcement_learning_assessment
python -m experiments.scalability_stress_test
python -m experiments.surrogate_gradient_validation
python -m experiments.test_directional_sensors
python -m experiments.test_evaluate_saved_checkpoint
python -m experiments.test_td_critic
python -m experiments.test_training_resume
```

Execute os comandos a partir da raiz do repositório e confirme que os arquivos correspondentes existem na versão local. Os resultados descritos neste documento pertencem aos logs fornecidos; novas execuções podem produzir valores diferentes.