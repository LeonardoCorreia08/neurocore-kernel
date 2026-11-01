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

# NEUROCORE

**Networked Engine for Unified Reasoning and Cognitive Operations**

Arquitetura experimental de inteligência artificial baseada em redes neurais pulsantes (*Spiking Neural Networks*, SNN).

## 1. Visão geral

O NEUROCORE é um projeto de pesquisa e desenvolvimento de uma arquitetura neural pulsante. O sistema usa neurônios artificiais, sinapses esparsas e aprendizado baseado no tempo dos disparos.

O projeto busca executar estas funções:

- Receber estímulos.
- Converter estímulos em pulsos neurais.
- Propagar pulsos pela rede.
- Associar padrões.
- Armazenar estados e pesos.
- Adaptar conexões com regras de plasticidade.
- Selecionar ações em um ambiente simulado.
- Medir atividade e desempenho.

O núcleo de processamento usa redes neurais pulsantes (*Spiking Neural Networks*, SNN). O projeto não usa Transformers ou LLMs como núcleo de inteligência.

> **Estado do projeto:** os registros fornecidos documentam testes de módulos, simulações com 10 mil neurônios e testes de alocação e execução de ciclos com 100 mil e 1 milhão de neurônios. Esses resultados, isoladamente, não demonstram inteligência geral, capacidade cognitiva validada em escala de um milhão de neurônios nem eficiência energética comprovada por medições de hardware.

## 2. Arquitetura em resumo

O sistema está organizado em cinco motores:

| Motor | Responsabilidade |
|---|---|
| Neural Engine | Simular neurônios LIF, sinapses e propagação de pulsos. |
| Memory Engine | Armazenar estados e associar padrões. |
| Learning Engine | Ajustar pesos com STDP, R-STDP e mecanismos homeostáticos. |
| Cognitive Engine | Integrar sinais, manter atenção e selecionar ações. |
| Evaluation Engine | Registrar atividade, desempenho e estimativas de custo. |

Os subsistemas funcionais incluem percepção, córtex sensorial, memória, aprendizado, cognição e ação motora.

Consulte [architecture.md](architecture.md) para conhecer os componentes e o fluxo de dados.

## 3. Resultados registrados

A tabela abaixo resume valores presentes nos logs fornecidos. Consulte [experiments_log.md](experiments_log.md) para detalhes, comandos e limitações de cada teste.

| Experimento | Resultado registrado |
|---|---|
| NeuroCore 0.1 — modelo base | 50 episódios; 6 metas atingidas acumuladas. |
| NeuroCore 0.1 — STDP | 50 episódios; 6 metas atingidas. |
| NeuroCore 0.1 — R-STDP | 50 episódios; 5 metas atingidas. |
| NeuroCore 0.2 | 30 episódios; 1 meta atingida; raster plot indicado em `docs/spike_raster.png`. |
| Rede de 10.000 neurônios | Construção da rede esparsa em 5,16 s; simulação de 50 passos em 17,5334 s; média de 4.585,72 spikes por passo. |
| Agente NeuroCore 0.3 | 1 meta atingida em 10 episódios. |
| NeuroCore 0.4 | 5 episódios de 80 passos; checkpoint salvo em `config/synapse_w1_checkpoint.pt`. |
| Subsistemas corticais | Ciclo de integração de 20 passos; recompensas exibidas iguais a 0,0 nos passos registrados. |
| Subsistemas de memória | Snapshot salvo e carregado; padrão `[0., 1., 0., 1.]` recuperado; 380 sinapses podadas e 1.416 ativas restantes. |
| Módulos de percepção | Codificadores de sensor, visão e áudio produziram vetores de pulsos de teste. |
| Monitoramento | 50 passos registrados; taxa de acerto de 80%; custo energético estimado de `0.00000047 J`. |
| Escala de 100.000 neurônios | Alocação em 2,02 s; ciclo de teste em 197,90 ms; 24.341 spikes ativos. |
| Escala de 1.000.000 de neurônios | Alocação em 29,87 s; ciclo de teste em 2.376,34 ms; 174.005 spikes ativos. |
| Generalização | Eficiência registrada de 70% no Ambiente A e taxa de sucesso de 100% no Ambiente B, conforme o teste. |
| Simulação dinâmica com 1 milhão de neurônios | Cabeçalho anuncia 1.000 passos, mas o relatório registra 50 passos; tempo total de 50,28 s. |

Os números são os valores impressos pelos testes disponibilizados. Eles não devem ser interpretados como resultados reproduzidos de forma independente. Os custos energéticos impressos pelos módulos são estimativas, salvo indicação expressa de medição; o log dinâmico registra `0.000000 Joules`, valor que não comprova consumo físico nulo.

## 4. Modelos matemáticos

O modelo neuronal de referência é o **Leaky Integrate-and-Fire (LIF)**.

O aprendizado local usa **Spike-Timing-Dependent Plasticity (STDP)**. Os experimentos também incluem uma variante de aprendizado modulada por recompensa (R-STDP).

Consulte [math_models.md](math_models.md) para ver as equações e as definições dos parâmetros.

## 5. Tecnologias registradas no projeto

| Tecnologia | Uso previsto ou registrado |
|---|---|
| Python 3.10+ | Linguagem principal declarada pelo projeto. |
| PyTorch | Tensores e matrizes esparsas, conforme os logs. |
| snnTorch | Prototipagem de componentes SNN; confirmar uso no código. |
| Brian2 | Simulação neural detalhada; confirmar uso no código. |
| NumPy | Operações numéricas; confirmar dependências e uso. |
| CUDA | Aceleração em GPU NVIDIA; confirmar configuração efetiva do ambiente. |
| Matplotlib | Geração de gráficos, conforme os logs. |
| HDF5 | Persistência de dados prevista; confirmar implementação. |
| Intel Lava | Integração neuromórfica futura. |

A presença de uma tecnologia nesta lista não confirma que todos os módulos já a utilizam. Verifique o código e o arquivo de dependências antes de instalar ou executar o projeto.

## 6. Estrutura de documentação

```text
.
├── README.md
├── architecture.md
├── experiments_log.md
└── math_models.md
```

O arquivo `experiments_log.md` documenta os resultados dos testes. Confirme se todos os arquivos e diretórios citados existem na sua cópia do repositório.

## 7. Execução

Os logs fornecidos mostram comandos de teste executados no ambiente Windows. A estrutura e os nomes dos módulos devem corresponder aos arquivos presentes na versão local do repositório.

### 7.1 Criar e ativar o ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências, caso o repositório contenha `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### 7.2 Executar os testes

Execute os comandos a partir da raiz do repositório. Os nomes abaixo aparecem nos logs fornecidos:

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
```

Execute apenas os comandos correspondentes aos arquivos existentes na sua cópia do repositório. Os nomes de módulos registrados nos logs não garantem que esses arquivos estejam presentes na versão atual.

## 8. Limites dos resultados atuais

Os testes disponíveis documentam a execução de componentes e testes de escalabilidade estrutural em condições específicas. Eles não demonstram todos os objetivos descritos no projeto.

Em particular:

- A alocação de um milhão de neurônios não comprova aprendizagem útil nessa escala.
- Os logs não informam a especificação completa do hardware, a versão exata do PyTorch ou todas as condições de carga.
- Os custos energéticos indicados pelos módulos de avaliação são estimativas; o log dinâmico registra zero joules, sem informações suficientes para validar uma medição física.
- Os resultados de metas não incluem uma definição completa da métrica nem um protocolo de repetição.
- O teste de memória associativa recuperou o padrão `[0., 1., 0., 1.]` no log fornecido. Isso documenta a saída desse teste, mas não comprova robustez da memória em condições variadas.
- Os logs registram avisos do PyTorch sobre verificações de invariantes de tensores esparsos desativadas.
- O teste de generalização registra 100% de sucesso no Ambiente B, mas não documenta detalhes suficientes sobre número de tentativas, sementes e diversidade dos ambientes para concluir que há generalização ampla.
- O teste dinâmico anuncia 1.000 passos, mas o relatório final registra 50. Essa divergência precisa ser investigada.
- Resultados de execuções distintas não devem ser comparados como se viessem de um experimento controlado, especialmente quando densidade esparsa e outras condições variam.

## 9. Próximas prioridades técnicas

1. Registrar hardware, sistema operacional, versões das bibliotecas, sementes e configuração em cada execução.
2. Investigar os avisos e validar a construção e a atualização de pesos esparsos.
3. Repetir os testes de memória associativa com diferentes padrões e níveis de ruído.
4. Definir métricas reproduzíveis para sucesso, retenção e generalização.
5. Repetir os testes com sementes controladas e múltiplas execuções.
6. Separar resultados de alocação, simulação, aprendizagem e desempenho energético.
7. Validar as estimativas de energia por meio de um método de medição documentado.
8. Investigar por que o teste dinâmico registra 50 passos quando anuncia 1.000.
9. Documentar as condições dos testes de generalização e avaliar mais ambientes e tentativas.

## 10. Licença

O projeto declara usar a licença MIT. Confirme se o arquivo `LICENSE` está presente e se seu conteúdo corresponde à licença aplicada ao repositório.
