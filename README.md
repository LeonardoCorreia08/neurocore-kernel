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

Arquitetura experimental de inteligência artificial baseada em redes neurais pulsantes (SNN).

## 1. Visão geral

O NEUROCORE é um projeto de pesquisa e desenvolvimento de uma arquitetura neural pulsante. O sistema usa neurônios artificiais, sinapses esparsas e aprendizado baseado no tempo dos disparos.

O projeto busca executar estas funções:

- receber estímulos;
- converter estímulos em pulsos neurais;
- propagar pulsos pela rede;
- associar padrões;
- armazenar estados e pesos;
- adaptar conexões com regras de plasticidade;
- selecionar ações em um ambiente simulado;
- medir atividade e desempenho.

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

Os registros de execução fornecidos incluem os seguintes resultados:

| Experimento | Resultado registrado |
|---|---|
| Rede de 10.000 neurônios | Matriz esparsa construída em 5,45 s. Simulação de 50 passos concluída em 17,4204 s. |
| Agente NeuroCore 0.3 | 1 meta atingida em 10 episódios. Uma execução anterior falhou durante a atualização de pesos esparsos. |
| NeuroCore 0.4 | 5 episódios executados. Checkpoint sináptico salvo. |
| Módulos corticais | 181 metas acumuladas em 50 episódios, conforme o log. |
| Módulos de memória | Persistência e recuperação de checkpoint testadas. 330 sinapses fracas removidas no teste de poda. |
| Módulos de percepção | Codificadores de sensor, visão e áudio produziram pulsos de teste. |
| Monitoramento | Teste concluído após correções. Taxa de acerto registrada: 80%. |
| Escala de 100.000 neurônios | Alocação concluída em 1,92 s. Ciclo de teste: 174,40 ms. |
| Escala de 1.000.000 de neurônios | Alocação concluída em 29,25 s. Ciclo de teste: 2.394,65 ms. |

Os valores são os valores impressos pelos testes. Consulte [experiment_logs.md](experiment_logs.md) para ver as condições e as limitações de cada resultado.

## 4. Modelos matemáticos

O modelo neuronal de referência é o **Leaky Integrate-and-Fire (LIF)**.

O aprendizado local usa **Spike-Timing-Dependent Plasticity (STDP)**. Os experimentos também incluem uma variante de aprendizado modulada por recompensa (R-STDP).

Consulte [math_models.md](math_models.md) para ver as equações e as definições dos parâmetros.

## 5. Tecnologias registradas no projeto

| Tecnologia | Uso previsto ou registrado |
|---|---|
| Python 3.10+ | Linguagem principal. |
| PyTorch | Tensores e matrizes esparsas. |
| snnTorch | Prototipagem de componentes SNN. |
| Brian2 | Simulação neural detalhada. |
| NumPy | Operações numéricas. |
| CUDA | Aceleração em GPU NVIDIA. |
| Matplotlib | Gráficos e visualização de atividade. |
| HDF5 | Persistência de dados, conforme a arquitetura planejada. |
| Intel Lava | Integração neuromórfica futura. |

A presença de uma tecnologia na lista não confirma que todos os módulos já a utilizam. Verifique o código e o arquivo de dependências antes de instalar ou executar o projeto.

## 6. Estrutura de documentação

```text
.
├── README.md
├── architecture.md
├── experiment_logs.md
└── math_models.md
```

## 7. Execução

Os logs fornecidos mostram comandos de teste executados no ambiente Windows. A estrutura e os nomes dos módulos podem mudar entre versões.

Ative o ambiente virtual e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Execute os testes a partir da raiz do repositório. Por exemplo:

```powershell
python -m experiments.test_05_scale_10k
python -m experiments.test_06_neurocore_03_agent
python -m experiments.neurocore_04_test
python -m experiments.test_09_memory_modules
python -m experiments.test_10_perception_modules
python -m experiments.test_11_monitoring
python -m experiments.test_12_scale_100k
python -m experiments.test_13_scale_1m
python -m experiments.test_14_generate_plots
```

Use apenas os comandos que correspondem aos arquivos existentes na sua cópia do repositório.

## 8. Limites dos resultados atuais

Os testes disponíveis demonstram execução de componentes e escalabilidade estrutural em condições específicas. Eles não demonstram todos os objetivos descritos no projeto.

Em particular:

- a alocação de um milhão de neurônios não comprova aprendizagem útil nessa escala;
- os logs não informam o hardware completo, a versão do PyTorch ou as condições de carga;
- os valores de energia são estimativas impressas pelo módulo de avaliação;
- os resultados de metas não incluem, nos logs fornecidos, uma definição completa da métrica ou um protocolo de repetição;
- o teste de memória associativa recuperou um vetor zerado. Esse resultado exige análise antes de afirmar que a associação foi validada;
- os logs registram avisos sobre verificações de invariantes de tensores esparsos desativadas.

## 9. Próximas prioridades técnicas

1. Registrar hardware, sistema operacional, versões e configuração em cada execução.
2. Corrigir e testar a atualização de pesos esparsos.
3. Verificar a recuperação de padrões na memória associativa.
4. Definir métricas reproduzíveis para sucesso, retenção e generalização.
5. Repetir os testes com sementes controladas e múltiplas execuções.
6. Separar resultados de alocação, simulação, aprendizagem e desempenho energético.
7. Comparar a estimativa de energia com medições de hardware.

## 10. Licença

O projeto usa a licença MIT. Confirme se o arquivo `LICENSE` está presente e atualizado no repositório.
