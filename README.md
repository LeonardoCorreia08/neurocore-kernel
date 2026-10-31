<div align="center">
  <img src="Assets/brain.png" alt="NEUROCORE Architecture Banner" width="280" />
  <h1>NEUROCORE</h1>
  <p><strong>Networked Engine for Unified Reasoning and Cognitive Operations</strong></p>
  <p><em>Arquitetura Cognitiva Experimental Baseada em Redes Neurais Pulsantes (SNN Pura)</em></p>
  <p>
    <img src="https://img.shields.io/badge/Arquitetura-SNN%20Pura%20%2F%20Neurom%C3%B3rfica-0052CC?style=flat-square" alt="Arquitetura" />
    <img src="https://img.shields.io/badge/Modelo%20Neuronal-LIF-00875A?style=flat-square" alt="LIF" />
    <img src="https://img.shields.io/badge/Aprendizado-STDP%20%2B%20Refor%C3%A7o-FF8B00?style=flat-square" alt="STDP" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square" alt="Python" />
    <img src="https://img.shields.io/badge/Doc%20Standard-ASD--STE100-success?style=flat-square" alt="ASD-STE100" />
  </p>
</div>

---

## 1. Visão Geral do Sistema (Descrição)

O **NEUROCORE** é um sistema de inteligência artificial inspirado na organização funcional de um cérebro biológico. O sistema percebe estímulos, aprende com experiências, forma memórias, reconhece padrões e adapta o seu comportamento.

Esta arquitetura utiliza exclusivamente **Redes Neurais Pulsantes (SNN)** orientadas a eventos. O sistema não utiliza *Transformers* ou Grandes Modelos de Linguagem (LLMs) como núcleo de inteligência. O desenvolvimento ocorre de forma progressiva: a rede inicia com $1.000$ neurônios e escala até $1.000.000$ de neurônios, medindo cada capacidade cognitiva experimentalmente.

### 1.1 Os Cinco Motores Fundamentais da Plataforma

| ID | Nome do Módulo | Função Principal | Componentes Técnicos |
| :--- | :--- | :--- | :--- |
| **Módulo 1** | **Neural Engine** | Simula neurônios biológicos, sinapses dinâmicas e propaga impulsos elétricos. | Neurônios *Leaky Integrate-and-Fire* (LIF), atrasos sinápticos e codificadores de pulsos. |
| **Módulo 2** | **Memory Engine** | Armazena dados em múltiplas escalas de tempo e associa experiências. | Memória sensorial, curto prazo, longo prazo e memória associativa. |
| **Módulo 3** | **Learning Engine** | Ajusta conexões neurais com base no tempo dos pulsos e em recompensas. | STDP, plasticidade homeostática, aprendizado por reforço e contínuo. |
| **Módulo 4** | **Cognitive Engine** | Filtra sinais relevantes, reconhece padrões e seleciona ações. | Mecanismos de atenção, córtex associativo e tomada de decisão. |
| **Módulo 5** | **Evaluation Engine** | Mede o aprendizado, a atividade neural e a eficiência energética. | Métricas científicas, telemetria de consumo e comparação com redes tradicionais. |

### 1.2 Subsistemas Funcionais do Cérebro Artificial

O núcleo compartilhado (**NEURAL CORE**) fornece neurônios pulsantes, sinapses, dinâmica temporal e plasticidade para seis subsistemas integrados:

1. **PERCEPTION ENGINE:** Captura visão, áudio, sinais de sensores e dados temporais.
2. **SENSORY CORTEX:** Executa o processamento sensorial primário dos impulsos.
3. **MEMORY ENGINE:** Gerencia a retenção e a associação de experiências.
4. **LEARNING ENGINE:** Controla a plasticidade sináptica e as regras de adaptação.
5. **COGNITIVE ENGINE:** Opera o foco de atenção e a tomada de decisão.
6. **MOTOR ENGINE:** Executa ações, respostas e interações diretas com o ambiente.

---

## 2. Arquitetura Neural

A unidade fundamental de processamento do NEUROCORE é o neurônio artificial baseado no modelo **Leaky Integrate-and-Fire (LIF)**.

O neurônio LIF executa cinco operações sequenciais:
1. Recebe impulsos elétricos (*spikes*) de outros neurônios conectados.
2. Acumula potencial elétrico na membrana.
3. Perde parte desse potencial gradualmente ao longo do tempo (vazamento).
4. Dispara um impulso elétrico quando o potencial ultrapassa o limiar de ativação.
5. Reinicia o potencial da membrana para o estado de repouso após o disparo.

### 2.1 Equação Diferencial do Neurônio LIF

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R I$$

Onde:
* $V$: Potencial atual da membrana.
* $\tau_m$: Constante de tempo da membrana.
* $V_{rest}$: Potencial elétrico de repouso.
* $R$: Resistência elétrica da membrana.
* $I$: Corrente elétrica total recebida pelas sinapses.

### 2.2 Especialização Funcional dos Neurônios

O sistema utiliza o mesmo modelo matemático LIF como base e altera apenas os parâmetros, pesos e topologias para criar seis tipos funcionais:

| Tipo de Neurônio | Função no Sistema |
| :--- | :--- |
| **Excitatório** | Aumenta o potencial elétrico e a atividade dos neurônios de destino. |
| **Inibitório** | Reduz o potencial elétrico e estabiliza a atividade da rede. |
| **Sensorial** | Converte estímulos externos brutos em trens de pulsos (*spikes*). |
| **Associativo** | Combina sinais provenientes de diferentes regiões sensoriais e de memória. |
| **Modulador** | Altera a taxa de aprendizado local por meio de sinais de recompensa. |
| **Motor** | Converte os disparos de decisão em comandos de ação no ambiente. |

---

## 3. Sistema de Sinapses e Plasticidade (STDP)

Cada conexão sináptica do NEUROCORE possui cinco propriedades configuráveis:
* **Peso sináptico ($w$):** Define a intensidade da conexão entre dois neurônios.
* **Atraso sináptico ($d$):** Define o tempo de propagação do impulso até o neurônio de destino.
* **Tipo de conexão:** Define o sinal da corrente (excitatória ou inibitória).
* **Plasticidade:** Habilita ou bloqueia a alteração dinâmica do peso sináptico.
* **Estado sináptico:** Registra o histórico temporal recente de disparos.

### 3.1 Regra Matemática da STDP

O sistema utiliza a **Plasticidade Dependente do Tempo de Disparo (STDP)** como mecanismo primário de aprendizado local. A diferença temporal ($\Delta t = t_{pos} - t_{pre}$) entre o disparo do neurônio pré-sináptico ($A$) e do neurônio pós-sináptico ($B$) determina a variação do peso ($\Delta w$):

$$\Delta w = \begin{cases} A_+ e^{-\Delta t / \tau_+}, & \Delta t > 0 \\ -A_- e^{\Delta t / \tau_-}, & \Delta t < 0 \end{cases}$$

* **Fortalecimento (LTP):** Se o neurônio $A$ dispara antes do neurônio $B$ ($\Delta t > 0$), a sinapse fica mais forte.
* **Enfraquecimento (LTD):** Se o neurônio $B$ dispara antes do neurônio $A$ ($\Delta t < 0$), a sinapse fica mais fraca.

### 3.2 Três Mecanismos Combinados de Estabilidade e Aprendizado

A regra STDP isolada não garante estabilidade nem aprendizado orientado a metas. O NEUROCORE combina três mecanismos simultâneos:

| Mecanismo | Objetivo Técnico |
| :--- | :--- |
| **STDP Clássica** | Aprender correlações e associações temporais entre eventos. |
| **Plasticidade Homeostática** | Regula o limiar de disparo para manter a atividade neural estável e evitar saturação. |
| **STDP Modulada por Recompensa** | Fortalece os caminhos sinápticos que geraram ações bem-sucedidas no ambiente. |

---

## 4. Arquitetura de Memória Artificial

O NEUROCORE implementa quatro camadas de memória inspiradas em circuitos córtico-hipocampais, utilizando redes recorrentes, grupos neurais especializados e reativação de padrões:

1. **Memória Sensorial:** Retém sinais brutos por curtíssimo prazo (ex.: rastrear o movimento imediato de um objeto diante de um sensor).
2. **Memória de Curto Prazo:** Mantém padrões neurais ativos por meio de loops recorrentes (ex.: lembrar a posição de um obstáculo que saiu temporariamente do campo de visão).
3. **Memória de Longo Prazo:** Consolida padrões nos pesos estruturais das sinapses para reconhecimento permanente em diferentes ocasiões.
4. **Memória Associativa:** Conecta estímulos multimodais distintos (imagem + som + ação executada + consequência obtida).

---

## 5. Ciclo de Aprendizado Autônomo

O NEUROCORE aprende explorando o ambiente sem depender de rótulos supervisionados para cada entrada e sem utilizar *backpropagation* tradicional como motor principal.

### 5.1 As Sete Etapas do Ciclo Cognitivo
1. **Observar:** Captura sinais dos sensores no ambiente.
2. **Processar:** Codifica os sinais brutos em impulsos neurais (*spikes*).
3. **Associar:** Correlaciona os pulsos atuais com memórias ativas.
4. **Experimentar:** Seleciona e executa uma ação motora no ambiente controlado.
5. **Avaliar:** Recebe um sinal de recompensa ou penalidade do ambiente.
6. **Aprender:** Aplica STDP modulada por recompensa para ajustar as sinapses ativas.
7. **Consolidar:** Preserva conexões úteis e poda associações irrelevantes (esquecimento ativo).

### 5.2 Modalidades de Aprendizado Suportadas
* **A. Aprendizado Não Supervisionado:** Extrai regularidades estatísticas e temporais sem rótulos externos.
* **B. Aprendizado por Reforço:** Otimiza políticas de ação por meio de sinais de dopamina simulada (recompensa).
* **C. Aprendizado Contínuo:** Acumula novas habilidades ao longo do tempo e mitiga o esquecimento catastrófico.

---

## 6. Stack Tecnológica e Ferramentas

| Tecnologia | Responsabilidade no Sistema |
| :--- | :--- |
| **Python 3.10+** | Linguagem principal de desenvolvimento e orquestração. |
| **PyTorch** | Operações tensoriais esparsas e execução em GPU. |
| **snnTorch** | Prototipagem rápida de camadas SNN e codificadores de pulsos. |
| **Brian2** | Simulação biológica detalhada de dinâmicas neurais e sinápticas. |
| **NumPy** | Processamento numérico e manipulação de matrizes. |
| **CUDA** | Aceleração paralela em placas de vídeo NVIDIA. |
| **Matplotlib** | Geração de *raster plots* e visualização de atividade neural. |
| **HDF5** | Armazenamento persistente de estados sinápticos e memórias. |
| **Intel Lava (Futuro)** | Mapeamento posterior para processadores neuromórficos (ex.: Intel Loihi 2). |

---

## 7. Plano de Escalabilidade (De $10^3$ a $10^6$ Neurônios)

Para atingir a escala de um milhão de neurônios sem reescrever o sistema, a arquitetura proíbe conexões densas completas (*all-to-all*) e adota **conectividade esparsa**, **processamento orientado a eventos** e **particionamento modular**.

| Etapa | Escala (Neurônios) | Foco de Engenharia e Validação |
| :--- | :--- | :--- |
| **Etapa 1** | $1.000$ | Validar dinâmica LIF, sinapses excitatórias/inibitórias, STDP e homeostase. |
| **Etapa 2** | $10.000$ | Introduzir circuitos recorrentes, memória de curto prazo e associação. |
| **Etapa 3** | $100.000$ |Separar córtices especializados e validar o aprendizado contínuo. |
| **Etapa 4** | $1.000.000$ | Implementar matrizes esparsas distribuídas em GPU e otimização de eventos. |

---

## 8. Estrutura de Diretórios do Projeto

A organização modular isola o núcleo matemático, as regiões cerebrais, as regras de plasticidade e o ambiente virtual:

* `neurocore/`
  * `assets/`
    * `brain.png` — Logotipo e diagrama da arquitetura do sistema.
  * `core/` — **Núcleo Neural (Módulo 1):**
    * `neuron.py` — Implementação do modelo LIF e tipos funcionais.
    * `synapse.py` — Pesos, atrasos, estados e matrizes esparsas.
    * `spike_engine.py` — Motor de propagação de eventos e pulsos.
    * `neural_network.py` — Montagem de populações e topologias.
    * `simulation.py` — Loop de passo de tempo ($\Delta t$) da simulação.
  * `brain/` — **Subsistemas Corticais (Módulos 2 e 4):**
    * `sensory_cortex.py` — Processamento sensorial primário.
    * `associative_cortex.py` — Integração multimodal de sinais.
    * `memory_engine.py` — Controlador central de memória.
    * `attention_engine.py` — Filtro de relevância e foco neural.
    * `decision_engine.py` — Seleção competitiva de ações.
    * `motor_engine.py` — Disparo de comandos para o ambiente.
  * `learning/` — **Motor de Aprendizado (Módulo 3):**
    * `stdp.py` — Regras locais de STDP exponencial.
    * `homeostasis.py` — Regulação de limiar e estabilidade.
    * `reward_learning.py` — Modulação sináptica por recompensa.
    * `continual_learning.py` — Consolidação e proteção de pesos.
  * `perception/` — **Codificadores de Entrada:**
    * `vision.py` — Conversão de quadros/eventos visuais em *spikes*.
    * `audio.py` — Conversão de frequências sonoras em *spikes*.
    * `sensor_encoder.py` — Codificação temporal e por taxa (*rate/temporal coding*).
  * `memory/` — **Camadas de Armazenamento:**
    * `short_term.py` — Buffers recorrentes de atividade.
    * `long_term.py` — Persistência estrutural de pesos.
    * `associative.py` — Grafos e matrizes de correlação.
    * `consolidation.py` — Rotinas de reforço e esquecimento.
  * `environment/` — **Simulador Interativo:**
    * `world.py` — Grade espacial com paredes, objetos e metas.
    * `simulator.py` — Física simplificada e ciclo de interação do agente.
    * `rewards.py` — Cálculo de sinais de recompensa e penalidade.
  * `monitoring/` — **Avaliação e Telemetria (Módulo 5):**
    * `spike_visualizer.py` — Gráficos de disparo (*raster plots*).
    * `activity_monitor.py` — Monitoramento de frequência e homeostase.
    * `performance.py` — Cálculo de métricas científicas e custo energético.
  * `experiments/` — **Bateria de Testes:**
    * `neuron_test.py` — Validação isolada de neurônios LIF.
    * `memory_test.py` — Testes de retenção e reativação de padrões.
    * `learning_test.py` — Execução do experimento *NeuroCore 0.1*.
  * `config/`
    * `brain_config.yaml` — Hiperparâmetros da rede ($V_{rest}, \tau_m, A_+, A_-$).
  * `main.py` — Ponto de entrada principal da plataforma.
  * `requirements.txt` — Lista de pacotes e dependências Python.

---

## 9. Primeiro Experimento Prático: NeuroCore 0.1

A versão **NeuroCore 0.1** valida a capacidade de aprendizado autônomo antes da adição de visão ou audição complexas.
[Ver Relatório de Experimentos](./docs/experiments_log.md)


### 9.1 Especificações da Versão 0.1
* **Escala Inicial:** $1.000$ neurônios simulados.
* **Modelo Neuronal:** *Leaky Integrate-and-Fire* (LIF) com populações excitatórias e inibitórias.
* **Regra de Aprendizado:** STDP local combinada com plasticidade homeostática e modulação por recompensa.
* **Ambiente Virtual:** Arena 2D contendo um agente artificial, paredes, obstáculos, objetos e alvos de recompensa.

### 9.2 Protocolo de Execução e Critério de Sucesso
1. O agente inicia na arena sem conhecimento prévio do mapa.
2. Os sensores de distância convertem a proximidade de paredes e alvos em *spikes*.
3. O agente executa movimentos exploratórios e recebe recompensas ao atingir os objetivos.
4. O sistema modifica as suas sinapses em tempo real durante a exploração.
5. **Critério de Sucesso:** O agente deve aumentar a sua taxa de alcance de metas e navegar com sucesso em mapas novos (não vistos no treinamento), sem utilizar regras pré-programadas.

---

## 10. Métricas Científicas de Avaliação (Módulo 5)

O módulo de avaliação mede oito indicadores quantitativos e compara o desempenho contra modelos de referência (*baselines*):

| Indicador | O Que Mede |
| :--- | :--- |
| **Taxa de Acerto** | Percentual de tarefas e objetivos concluídos com sucesso no ambiente. |
| **Plasticidade** | Taxa de variação ($\Delta w$) e convergência dos pesos sinápticos. |
| **Retenção** | Tempo de preservação de um padrão aprendido sem reexposição ao estímulo. |
| **Generalização** | Desempenho do agente em configurações inéditas do ambiente virtual. |
| **Adaptação** | Velocidade de ajuste sináptico após uma mudança nas regras do ambiente. |
| **Esquecimento** | Degradação de tarefas antigas após o aprendizado de novas tarefas. |
| **Eficiência** | Operações sinápticas por segundo (SOPs), uso de memória e consumo energético. |
| **Atividade Neural** | Frequência média (Hz) e esparsidade populacional dos *spikes*. |

---

## 11. Cronograma de Desenvolvimento (Roadmap)

*Estimativas de planejamento para desenvolvimento individual e progressivo:*

| Fase | Período Estimado | Entrega Técnica Principal | Status |
| :--- | :--- | :--- | :--- |
| **Fase 1: Fundamentos SNN** | 2–3 semanas | Neurônios LIF, sinapses e motor de eventos (`core/`). | `[ ]` |
| **Fase 2: Sinapses e STDP** | 3–4 semanas | Regras STDP, homeostase e aprendizado associativo básico. | `[ ]` |
| **Fase 3: Memória Artificial** | 3–5 semanas | Circuitos recorrentes, retenção e memória associativa. | `[ ]` |
| **Fase 4: Ambiente Virtual (v0.1)** | 2–4 semanas | Arena 2D interativa e validação do *NeuroCore 0.1* ($1.000$ neurônios). | `[ ]` |
| **Fase 5: Arquitetura Cognitiva** | 4–8 semanas | Módulos de atenção, córtex associativo e tomada de decisão. | `[ ]` |
| **Fase 6: Escala de $100.000$** | 4–8 semanas | Especialização regional, matrizes esparsas e aprendizado contínuo. | `[ ]` |
| **Fase 7: Escala de $1.000.000$** | Variável | Otimização massiva em GPU e distribuição multi-dispositivo. | `[ ]` |

---

## 12. Requisitos de Hardware

Verifique se a estação de trabalho atende às especificações recomendadas para as Etapas 1 a 3 ($1.000$ a $100.000$ neurônios):

| Componente | Especificação Recomendada |
| :--- | :--- |
| **Processador (CPU)** | 8 núcleos físicos ou superior (`x86_64`). |
| **Memória RAM** | 32 GB DDR4/DDR5. |
| **Placa de Vídeo (GPU)** | NVIDIA RTX com 12 GB ou mais de VRAM (suporte a CUDA). |
| **Armazenamento** | 1 TB SSD NVMe (para gravação de estados sinápticos em HDF5). |
| **Sistema Operacional** | Linux (Ubuntu 22.04+) ou Windows 10/11 com ambiente Python. |

---

## 13. Procedimento de Instalação e Operação (Instruções)

Execute os passos abaixo em ordem sequencial para configurar e iniciar o **NeuroCore 0.1**:

1. Abra o terminal de linha de comando.
2. Clone o repositório para o seu disco local:
   `git clone https://github.com/seu-usuario/neurocore.git`
3. Acesse o diretório raiz do projeto:
   `cd neurocore`
4. Crie um ambiente virtual Python isolado:
   `python -m venv .venv`
5. Ative o ambiente virtual:
   * **No Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
   * **No Linux / macOS:** `source .venv/bin/activate`
6. Instale todas as dependências listadas no projeto:
   `pip install -r requirements.txt`
7. Execute o teste unitário do núcleo LIF e STDP (Meta Inicial):
   `python -m experiments.neuron_test`
8. Inicie o experimento de aprendizado autônomo **NeuroCore 0.1** ($1.000$ neurônios):
   `python main.py --config config/brain_config.yaml --experiment v0.1`

---

## 14. Licença

Este projeto utiliza a Licença MIT. Consulte o arquivo `LICENSE` para ler os termos legais completos.
