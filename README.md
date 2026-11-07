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


## 1. Visão geral

O NEUROCORE é um projeto experimental de pesquisa e desenvolvimento voltado à exploração de arquiteturas computacionais inspiradas em princípios do funcionamento neural biológico.

A proposta consiste em investigar como mecanismos de comunicação por pulsos, plasticidade sináptica, memória associativa e adaptação neural podem ser combinados em uma arquitetura computacional orientada a processos cognitivos.

O projeto parte do estudo das Spiking Neural Networks (SNN), explorando uma abordagem na qual o processamento de informações acontece por meio de eventos neurais discretos, considerando o tempo dos disparos, a dinâmica das conexões e a interação entre populações neuronais.

A intenção é investigar uma arquitetura na qual processamento, associação, aprendizado e adaptação sejam construídos a partir de mecanismos neurais pulsantes.

**Estado do projeto:** Em desenvolvimento. A arquitetura, os mecanismos e as capacidades pretendidas ainda estão sujeitos a implementação, experimentação, revisão e validação.

## 2. Motivação

Grande parte das arquiteturas modernas de inteligência artificial utiliza redes profundas, processamento matricial e modelos baseados em Transformers. O NEUROCORE parte de outra linha de investigação: explorar modelos computacionais inspirados na dinâmica dos sistemas nervosos biológicos.

A pesquisa busca compreender como mecanismos neurais locais podem contribuir para a construção de sistemas adaptativos, capazes de processar estímulos, modificar conexões, associar informações e desenvolver representações por meio da atividade neural.

Entre as questões que motivam o projeto:
* Como construir sistemas computacionais baseados em comunicação neural por pulsos?
* Como utilizar a plasticidade sináptica para permitir adaptação?
* Como investigar memória associativa?
* Como integrar percepção, memória e aprendizado?
* Como desenvolver mecanismos de processamento distribuído?
* Quais são os limites de escalabilidade de arquiteturas baseadas em SNN?
* Como avaliar a eficiência computacional e energética dessas arquiteturas?

Essas questões representam direções de investigação, não capacidades já demonstradas pelo sistema.

## 3. Objetivos

O objetivo de longo prazo é explorar uma arquitetura neural experimental capaz de integrar diferentes mecanismos de processamento e adaptação.

As principais áreas de investigação incluem:
* Processamento de estímulos por codificação neural.
* Comunicação entre populações de neurônios artificiais.
* Formação e recuperação de padrões associativos.
* Aprendizado por plasticidade sináptica.
* Adaptação baseada em experiência.
* Integração de diferentes fontes de informação.
* Desenvolvimento de mecanismos de seleção de respostas.
* Avaliação de escalabilidade e desempenho computacional.

## 4. Fundamentos científicos

A arquitetura considera conceitos fundamentais da computação neuromórfica e das redes neurais pulsantes.

| Conceito | Aplicação investigada |
| :--- | :--- |
| **Spiking Neural Networks** | Base da arquitetura neural. |
| **Leaky Integrate-and-Fire** | Modelagem de neurônios artificiais. |
| **Spike-Timing-Dependent Plasticity** | Aprendizado dependente do tempo dos disparos. |
| **Reward-Modulated STDP** | Aprendizado modulado por recompensa. |
| **Synaptic Plasticity** | Adaptação das conexões neurais. |
| **Homeostasis** | Estabilidade da atividade neural. |
| **Associative Memory** | Armazenamento e recuperação de padrões. |
| **Sparse Connectivity** | Conectividade neural esparsa. |
| **Lateral Inhibition** | Competição entre populações neuronais. |
| **Neuromodulation** | Investigação de mecanismos de modulação do aprendizado. |

A implementação desses mecanismos ocorrerá progressivamente, conforme a evolução dos experimentos.

## 5. Arquitetura conceitual

A organização inicial do NEUROCORE considera cinco áreas principais de desenvolvimento. Essa divisão representa uma proposta arquitetural e poderá ser modificada conforme os resultados da pesquisa.

### 5.1 Neural Engine
Responsável pela investigação dos mecanismos fundamentais de processamento neural.
**Possíveis responsabilidades:**
* Modelagem de neurônios artificiais.
* Simulação de disparos neurais.
* Representação de potenciais de membrana.
* Conectividade sináptica.
* Propagação de eventos neurais.

### 5.2 Memory Engine
Área dedicada à investigação de mecanismos de memória associativa.
**Possíveis responsabilidades:**
* Associação de padrões.
* Armazenamento de estados.
* Recuperação de informações.
* Persistência de conexões.
* Avaliação de retenção e estabilidade.

### 5.3 Learning Engine
Área destinada à investigação dos mecanismos de aprendizado e adaptação.
**Possíveis responsabilidades:**
* Plasticidade sináptica.
* STDP e R-STDP.
* Modulação por recompensa.
* Ajuste de pesos.
* Mecanismos homeostáticos.

### 5.4 Cognitive Engine
Área experimental destinada à integração dos mecanismos da arquitetura.
**Possíveis responsabilidades:**
* Integração de informações.
* Associação entre estímulos e estados internos.
* Exploração de mecanismos de atenção.
* Seleção de respostas.
* Coordenação entre subsistemas.

### 5.5 Evaluation Engine
Área destinada à instrumentação e avaliação experimental.
**Possíveis responsabilidades:**
* Monitoramento de atividade neural.
* Coleta de métricas.
* Avaliação de desempenho.
* Análise de escalabilidade.
* Registro de experimentos.
* Reprodutibilidade dos resultados.

## 6. Estado do desenvolvimento

O NEUROCORE encontra-se em fase de pesquisa, construção e experimentação. Para manter a documentação transparente, o desenvolvimento utilizará três classificações:

| Estado | Significado |
| :--- | :--- |
| **Implementado** | Componente existente no código-fonte. |
| **Em desenvolvimento** | Componente cuja implementação está em andamento. |
| **Planejado** | Funcionalidade prevista, ainda não implementada ou validada. |

Os objetivos arquiteturais não devem ser confundidos com funcionalidades disponíveis. Experimentos, protótipos e resultados serão documentados separadamente.

## 7. Roadmap

O roadmap representa uma sequência inicial de desenvolvimento. As etapas não possuem prazos fixos e poderão ser reorganizadas conforme os resultados obtidos.

**Fase 1 — Fundamentos e estruturação**
* Consolidar os fundamentos matemáticos.
* Definir modelos neuronais de referência.
* Estruturar a organização inicial do projeto.
* Definir padrões de documentação.
* Preparar o ambiente de experimentação.

**Fase 2 — Núcleo neural**
* Desenvolver o modelo neuronal LIF.
* Investigar conectividade sináptica.
* Implementar propagação de spikes.
* Avaliar configurações de redes esparsas.
* Criar experimentos de validação.

**Fase 3 — Aprendizado e plasticidade**
* Investigar mecanismos STDP.
* Explorar aprendizado modulado por recompensa.
* Implementar mecanismos de adaptação sináptica.
* Investigar mecanismos homeostáticos.
* Avaliar estabilidade e retenção.

**Fase 4 — Memória associativa**
* Investigar mecanismos de associação.
* Desenvolver experimentos de recuperação de padrões.
* Avaliar resistência a ruído.
* Investigar persistência de estados.
* Avaliar retenção em diferentes condições.

**Fase 5 — Integração dos subsistemas**
* Integrar os mecanismos neurais.
* Investigar a interação entre memória e aprendizado.
* Explorar mecanismos de percepção.
* Desenvolver experimentos de integração.
* Avaliar seleção de respostas.

**Fase 6 — Avaliação e escalabilidade**
* Definir métricas experimentais.
* Avaliar desempenho computacional.
* Investigar escalabilidade.
* Comparar configurações de rede.
* Melhorar a reprodutibilidade.
* Documentar limitações e resultados.

## 8. Tecnologias e ferramentas

As tecnologias abaixo representam ferramentas consideradas para a pesquisa e o desenvolvimento.

| Tecnologia | Finalidade |
| :--- | :--- |
| **Python** | Linguagem de desenvolvimento. |
| **PyTorch** | Operações tensoriais e processamento numérico. |
| **NumPy** | Operações matemáticas. |
| **snnTorch** | Experimentação com redes neurais pulsantes. |
| **Brian2** | Simulação neural. |
| **CUDA** | Possível aceleração em GPU NVIDIA. |
| **Matplotlib** | Visualização de experimentos. |
| **HDF5** | Possível persistência de dados. |

A presença de uma tecnologia nesta lista não significa que ela já esteja integrada ao código-fonte.

## 9. Pesquisa e referências científicas

O desenvolvimento do NEUROCORE considera trabalhos científicos relacionados a redes neurais pulsantes, plasticidade sináptica, aprendizado neural e arquiteturas inspiradas no cérebro. As referências abaixo constituem uma base inicial de estudo, não uma declaração de que os mecanismos descritos já foram reproduzidos no projeto.

### 9.1 Spiking Neural Networks
1. **Deep Learning in Spiking Neural Networks (2019)**
   * Autores: Amirhossein Tavanaei, Masoud Ghodrati, Saeed Reza Kheradpisheh, Timothée Masquelier e Anthony Maida.
   * Uma revisão sobre métodos de treinamento de SNNs profundas, aprendizado supervisionado e não supervisionado e desafios relacionados à diferenciação dos disparos neurais.
   * Publicação: Neural Networks, Volume 111, páginas 47–63.
   * DOI: [https://doi.org/10.1016/j.neunet.2018.12.002](https://doi.org/10.1016/j.neunet.2018.12.002)

2. **Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity (2015)**
   * Autores: Peter U. Diehl e Matthew Cook.
   * Trabalho experimental sobre reconhecimento de padrões utilizando SNN, STDP, inibição lateral e limiar adaptativo. É uma referência relevante para investigar aprendizado não supervisionado e formação de representações por plasticidade sináptica.
   * Publicação: Frontiers in Computational Neuroscience.
   * DOI: [https://doi.org/10.3389/fncom.2015.00099](https://doi.org/10.3389/fncom.2015.00099)

3. **Surrogate Gradient Learning in Spiking Neural Networks (2019)**
   * Autores: Emre O. Neftci, Hesham Mostafa e Friedemann Zenke.
   * Apresenta métodos de treinamento de SNN utilizando gradientes substitutos, uma abordagem importante para investigar arquiteturas profundas e aprendizado baseado em otimização.
   * Publicação: IEEE Signal Processing Magazine.
   * DOI: [https://doi.org/10.1109/MSP.2019.2931595](https://doi.org/10.1109/MSP.2019.2931595)

### 9.2 Computação neuromórfica
4. **Neuro-Inspired Computing Chips (2020)**
   * Autores: Wenqiang Zhang e colaboradores.
   * Revisão sobre chips de computação inspirada no cérebro, incluindo arquiteturas neuromórficas, eficiência energética e capacidade de aprendizado local.
   * Publicação: Nature Electronics.
   * DOI: [https://doi.org/10.1038/s41928-020-0435-7](https://doi.org/10.1038/s41928-020-0435-7)

5. **Spike-Based Neuromorphic Computing: An Overview from Bio-Inspiration to Hardware Architectures and Learning Mechanisms (2026)**
   * Autores: Anteneh Gebregiorgis e colaboradores.
   * Revisão abrangente sobre modelos neuronais, mecanismos de aprendizado, arquiteturas neuromórficas e implementação em hardware.
   * Publicação: Microprocessors and Microsystems.
   * DOI: [https://doi.org/10.1016/j.micpro.2025.105240](https://doi.org/10.1016/j.micpro.2025.105240)

### 9.3 Arquiteturas pós-Transformer e BDH
O BDH não faz parte da arquitetura atual do NEUROCORE. Entretanto, representa uma linha de pesquisa potencialmente relevante para estudos futuros sobre memória sináptica, aprendizado Hebbiano, topologia de redes e processamento neural.

6. **The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain (2025)**
   * Autores: Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska e Michał Bartoszkiewicz.
   * O trabalho apresenta a arquitetura Dragon Hatchling (BDH), baseada em uma rede biologicamente inspirada, com interações neuronais locais, topologia scale-free e mecanismos de aprendizado Hebbiano.
   * Publicação: arXiv, setembro de 2025.
   * Identificador: [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
   
   *Observação: BDH não deve ser tratado como sinônimo de SNN convencional. Apesar de compartilhar conceitos de neurônios pulsantes e plasticidade sináptica, sua arquitetura e seus objetivos computacionais são diferentes.*

## 10. Direções futuras de pesquisa

Além do desenvolvimento inicial baseado em SNN, o NEUROCORE poderá investigar outras arquiteturas e abordagens relacionadas à computação inspirada no cérebro. Entre as possibilidades de pesquisa:
* Topologias neurais scale-free.
* Redes com conectividade modular.
* Aprendizado Hebbiano local.
* Memória de trabalho baseada em plasticidade sináptica.
* Integração de mecanismos STDP e aprendizado por recompensa.
* Comparação entre arquiteturas SNN e modelos pós-Transformer.
* Investigação de arquiteturas híbridas.
* Avaliação de formulações adequadas para execução em GPU.
* Possível integração futura de conceitos inspirados no BDH.

Essas possibilidades não representam decisões arquiteturais definitivas. Qualquer integração deverá ser precedida por estudo teórico, prototipagem e avaliação experimental.

## 11. Desafios técnicos em aberto

O desenvolvimento de uma arquitetura baseada em SNN apresenta desafios que precisam ser investigados:
* Eficiência da simulação de grandes populações neuronais.
* Gerenciamento de conectividade esparsa.
* Estabilidade da plasticidade sináptica.
* Retenção e recuperação de padrões.
* Interação entre diferentes mecanismos de aprendizado.
* Definição de métricas adequadas.
* Generalização em diferentes ambientes.
* Custo computacional e energético.
* Reprodutibilidade dos experimentos.

Esses desafios fazem parte da própria investigação e poderão influenciar as decisões arquiteturais futuras.

## 12. Estrutura prevista do repositório

A estrutura abaixo representa uma organização de referência, sujeita a modificações durante o desenvolvimento.

```text
NEUROCORE/
│
├── README.md
│
├── docs/
│   ├── VISION.md
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   ├── RESEARCH_NOTES.md
│   ├── EXPERIMENTS.md
│   └── OPEN_QUESTIONS.md
│
├── src/
│   ├── neural/
│   ├── memory/
│   ├── learning/
│   ├── cognitive/
│   └── evaluation/
│
├── experiments/
│
├── tests/
│
├── Assets/
│
├── config/
│
└── LICENSE
```

*Nem todos os diretórios e componentes estão necessariamente implementados.*

## 13. Princípios de desenvolvimento

O NEUROCORE seguirá alguns princípios orientadores:
* **Investigação antes de afirmação:** Hipóteses devem ser avaliadas por meio de experimentos.
* **Desenvolvimento incremental:** A arquitetura será construída progressivamente, permitindo validar componentes antes de integrá-los.
* **Documentação transparente:** Funcionalidades planejadas não serão apresentadas como implementadas.
* **Reprodutibilidade:** Experimentos devem registrar parâmetros e condições de execução sempre que possível.
* **Evolução arquitetural:** Decisões técnicas poderão ser revistas conforme novas evidências forem obtidas.

## 14. Participação e contribuições

O NEUROCORE é um projeto experimental em evolução. Discussões técnicas, sugestões de arquitetura, estudos relacionados a SNN e contribuições para a pesquisa são bem-vindos.

O objetivo é construir uma base de investigação que permita explorar novas possibilidades de arquiteturas neurais.

## 15. Licença

A licença do projeto será definida e documentada no arquivo `LICENSE`. Consulte esse arquivo para conhecer as condições de utilização, modificação e distribuição do código.

<br>
