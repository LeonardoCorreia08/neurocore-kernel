# Arquitetura do NEUROCORE

## 1. Objetivo

Este documento descreve a arquitetura funcional do NEUROCORE com base no README do projeto e nos registros de execução fornecidos.

O sistema usa uma arquitetura de redes neurais pulsantes (SNN). O modelo neuronal de referência é o LIF. A rede usa conectividade esparsa para reduzir o custo de armazenamento e processamento das conexões.

Este documento separa:

- **Arquitetura descrita:** componentes definidos no projeto.
- **Evidência de execução:** componentes que aparecem nos logs.
- **Planejamento:** componentes previstos, mas sem evidência suficiente nos logs.

## 2. Visão em camadas

```text
                 AMBIENTE / ESTÍMULOS
                         |
                         v
                PERCEPTION ENGINE
           sensores | visão | áudio
                         |
                         v
                  SENSORY CORTEX
                         |
                         v
                   NEURAL CORE
             neurônios LIF e sinapses
                         |
             +-----------+-----------+
             |                       |
             v                       v
       MEMORY ENGINE            LEARNING ENGINE
   curto prazo / longo prazo   STDP / R-STDP /
       associativa             homeostase
             |                       |
             +-----------+-----------+
                         |
                         v
                 COGNITIVE ENGINE
             atenção e decisão
                         |
                         v
                   MOTOR ENGINE
                         |
                         v
                  AMBIENTE / AÇÃO

                         |
                         v
                 EVALUATION ENGINE
          atividade | métricas | telemetria
```

O diagrama mostra o fluxo funcional proposto. Ele não afirma que todos os caminhos estão integrados em uma única execução.

## 3. Motores principais

### 3.1 Neural Engine

**Responsabilidade:** simular a dinâmica neural e a propagação de pulsos.

Componentes descritos no projeto:

- neurônios LIF;
- sinapses excitatórias e inibitórias;
- pesos sinápticos;
- atrasos de propagação;
- motor de eventos;
- topologias neurais;
- simulação em passos de tempo.

**Evidência nos logs:** construção de matrizes esparsas, simulação de 10 mil neurônios e ciclos de teste com 100 mil e 1 milhão de neurônios.

### 3.2 Memory Engine

**Responsabilidade:** manter estados e padrões aprendidos.

Camadas descritas:

1. Memória sensorial: mantém estímulos por um intervalo curto.
2. Memória de curto prazo: mantém padrões ativos por recorrência.
3. Memória de longo prazo: persiste informações nos pesos e nos estados armazenados.
4. Memória associativa: relaciona padrões de diferentes fontes.

**Evidência nos logs:**

- gravação e leitura de um checkpoint;
- recuperação de um estado com pesos de amostra e época;
- teste de poda de sinapses;
- teste de memória associativa.

**Limite:** o teste associativo registrou a recuperação de um vetor com quatro valores iguais a zero. O log não demonstra que o padrão correto foi recuperado. Investigue esse teste antes de declarar a memória associativa validada.

### 3.3 Learning Engine

**Responsabilidade:** adaptar as conexões da rede.

Mecanismos descritos:

- STDP;
- STDP modulada por recompensa (R-STDP);
- plasticidade homeostática;
- consolidação;
- esquecimento ativo.

**Evidência nos logs:** execução de um agente com R-STDP, salvamento de checkpoint e poda de sinapses fracas.

**Limite:** os logs não apresentam uma curva de aprendizagem controlada que isole o efeito de cada regra.

### 3.4 Cognitive Engine

**Responsabilidade:** integrar sinais, selecionar informações relevantes e escolher ações.

Componentes descritos:

- atenção;
- córtex associativo;
- decisão;
- seleção de ações.

**Evidência nos logs:** existe um teste denominado `neurocore_04_test`, descrito como atenção cognitiva, obstáculos e persistência. Os logs também mostram experimentos com módulos corticais.

**Limite:** os registros não especificam o algoritmo de atenção nem uma métrica isolada de decisão.

### 3.5 Evaluation Engine

**Responsabilidade:** medir a atividade e o desempenho.

Indicadores previstos:

- taxa de sucesso;
- variação dos pesos;
- retenção;
- generalização;
- adaptação;
- esquecimento;
- operações sinápticas;
- atividade neural;
- estimativa de energia.

**Evidência nos logs:** cálculo de frequência média, desvio homeostático, taxa de acerto, operações sinápticas e estimativa de energia. O teste de monitoramento foi concluído após correções de sintaxe e importação.

## 4. Subsistemas de entrada e saída

### 4.1 Percepção

O projeto descreve codificadores para:

- sensores;
- visão;
- áudio.

Os codificadores convertem valores de entrada em pulsos. O teste de percepção registrou saídas para as três modalidades e propagação dos pulsos para uma camada LIF.

O log é uma verificação funcional com vetores de teste. Ele não demonstra desempenho em dados reais de visão ou áudio.

### 4.2 Ambiente e ação

O ambiente experimental descrito é uma arena 2D. Ele contém um agente, obstáculos, paredes, objetos e metas.

O agente recebe sinais dos sensores e seleciona movimentos. O ambiente retorna recompensas ou penalidades.

Os logs registram episódios e metas atingidas. Eles não fornecem todos os detalhes do mapa, da política de ação ou da distribuição das metas.

## 5. Conectividade esparsa e escalabilidade

O projeto evita uma matriz totalmente conectada (*all-to-all*). Os logs mostram as seguintes densidades:

| Escala | Densidade registrada | Tempo de construção |
|---:|---:|---:|
| 10.000 neurônios | 15% | 5,45 s |
| 100.000 neurônios | 5% | 1,92 s |
| 1.000.000 de neurônios | 1% | 29,25 s |

As densidades mudam entre os testes. Portanto, os tempos não formam uma comparação direta de desempenho entre escalas.

### 5.1 Ciclos de simulação registrados

| Escala | Tempo do ciclo | Spikes ativos | SOPs |
|---:|---:|---:|---:|
| 10.000 | 17,4204 s para 50 passos | média de 4.577,72 por passo | Não informado |
| 100.000 | 174,40 ms | 26.094 | 5.781.512 |
| 1.000.000 | 2.394,65 ms | 166.992 | 69.163.386 |

Os testes de 100 mil e 1 milhão registram um ciclo de teste. Os logs não informam que esses ciclos executaram a mesma quantidade de passos ou a mesma carga. Não compare os tempos como se fossem uma medição normalizada.

## 6. Persistência

Os logs mostram um checkpoint salvo em:

```text
config/synapse_w1_checkpoint.pt
```

O teste de memória também salvou e carregou:

```text
memory/storage/neurocore_snapshot_test.pt
```

Esses caminhos são evidências dos testes. Confirme a estrutura atual do repositório antes de usá-los em novos scripts.

## 7. Fluxo de execução

O ciclo funcional proposto é:

1. O ambiente fornece estímulos.
2. Os codificadores convertem os estímulos em pulsos.
3. O Neural Core propaga os pulsos.
4. A memória fornece estados e padrões anteriores.
5. O mecanismo de decisão seleciona uma ação.
6. O ambiente executa a ação.
7. O ambiente retorna uma recompensa ou penalidade.
8. O Learning Engine ajusta as sinapses.
9. O sistema registra as métricas.

## 8. Estrutura de módulos descrita

A estrutura abaixo resume os diretórios definidos no README original. Confirme os nomes com a árvore atual do repositório.

```text
neurocore/
├── core/
│   ├── neuron.py
│   ├── synapse.py
│   ├── spike_engine.py
│   ├── neural_network.py
│   └── simulation.py
├── brain/
│   ├── sensory_cortex.py
│   ├── associative_cortex.py
│   ├── memory_engine.py
│   ├── attention_engine.py
│   ├── decision_engine.py
│   └── motor_engine.py
├── learning/
│   ├── stdp.py
│   ├── homeostasis.py
│   ├── reward_learning.py
│   └── continual_learning.py
├── perception/
│   ├── vision.py
│   ├── audio.py
│   └── sensor_encoder.py
├── memory/
│   ├── short_term.py
│   ├── long_term.py
│   ├── associative.py
│   └── consolidation.py
├── environment/
│   ├── world.py
│   ├── simulator.py
│   └── rewards.py
├── monitoring/
│   ├── spike_visualizer.py
│   ├── activity_monitor.py
│   └── performance.py
├── experiments/
├── config/
├── main.py
└── requirements.txt
```

## 9. Limites conhecidos e pontos de verificação

- O PyTorch emitiu avisos porque as verificações de invariantes dos tensores esparsos estavam desativadas.
- Um teste de R-STDP falhou ao somar tensores com dimensões incompatíveis. Uma execução posterior terminou, mas o log não documenta a correção.
- O teste de monitoramento falhou primeiro por erro de sintaxe e depois por falha de importação. Uma execução posterior passou.
- O teste de memória associativa retornou um vetor zerado.
- A escala de um milhão foi testada em alocação e em um ciclo de propagação. O log não demonstra aprendizagem nessa escala.

## 10. Próximos passos de arquitetura

1. Definir contratos de entrada e saída para cada módulo.
2. Documentar dimensões e formatos dos tensores.
3. Validar os índices e valores das matrizes esparsas.
4. Separar testes de unidade, integração e escala.
5. Criar um protocolo comum para os testes de 10 mil, 100 mil e 1 milhão de neurônios.
6. Registrar a versão da arquitetura em cada checkpoint.
7. Definir como a memória associativa mede uma recuperação correta.
