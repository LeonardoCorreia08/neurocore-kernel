# Revisão dos Experimentos 01–16

Esta revisão compara os scripts do projeto com o log fornecido e ajusta chamadas incompatíveis, parâmetros de configuração e critérios que produziam resultados enganosos.

| Teste | Resultado da revisão |
|---|---|
| 01 — LIF base | Mantido. Adicionados asserts para integração, disparo, leak e reset. A execução do ambiente agora é descrita como simulação base, não como aprendizagem, pois não há plasticidade neste script. |
| 02 — STDP | O arquivo desta cópia já usa `compute_sparse_deltas` e não contém a chamada antiga `update_weights` mostrada no log. O log parece vir de uma versão diferente/desatualizada do arquivo. |
| 03 — R-STDP | Passa os índices `[pós, pré]` exigidos a `step_eligibility`; aplica os deltas 1D retornados por `apply_reward_modulation` nas posições correspondentes das matrizes densas. |
| 04 — NeuroCore 0.2 | Corrigidas as chamadas de R-STDP para fornecer índices e aplicar deltas conforme a API atual. O raster é salvo pelo visualizador, que cria o diretório. |
| 05 — Escala 10k | Substituído o acesso removido `config['scale']` pelo perfil de migração `legacy.scale`; corrigido o nome de parâmetro do LIF para `decay`. |
| 06 — Agente 10k | Migrado para `legacy.scale`; as chamadas R-STDP agora passam índices coalescidos e atualizam as sinapses esparsas com os deltas alinhados. |
| 07 — NeuroCore 0.4 | Migrado para `legacy.scale`; alinhadas as chamadas R-STDP e substituída a chamada inexistente de persistência por serialização explícita dos pesos esparsos. |
| 08 — Módulos corticais | Migrado para `legacy.scale`; após o reteste revelar ação 0 em todos os passos, o estímulo atencional foi amplificado e o script agora registra spikes de decisão e contagem de ações, sem declarar diversidade comportamental automaticamente. Essa alteração ainda deve ser reexecutada. |
| 09 — Memória associativa | Substituído o critério “qualquer spike significa memória” por padrão determinístico, pesos associativos Hebbianos e verificação de recuperação exata em cinco ciclos. |
| 10 — Percepção | Sem alteração: o log registra execução concluída e as dimensões observadas são coerentes com os módulos. |
| 11 — Monitoramento | Sem alteração: o log registra execução concluída; energia aparece arredondada como estimativa, não como medição exata. |
| 12 — Escala 100k | Sem alteração: o log registra conclusão. Não foi repetido por ser benchmark de escala. |
| 13 — Escala 1M | Corrigida a ordem de definição de `output_dim`/`hidden_dim` e a soma agora totaliza exatamente 1.000.000 neurônios. Não foi executado. |
| 14 — Gráficos | Sem alteração: o log registra que os três gráficos foram gerados. |
| 15 — Generalização | Removidos percentuais artificiais que eram incrementados por condições fixas, não por alcance de metas. Agora mede fração de atividade neural nos domínios A e B com pesos congelados e declara que isso não prova sucesso em tarefa. |
| 16 — Dinâmica 1M | A saída distingue energia reportada por NVML de NVML indisponível; `0.000000 J` não deve ser interpretado automaticamente como consumo físico zero. Não foi executado. |

## Validação realizada

Todos os scripts `experiments/test_*.py` foram compilados com `py_compile`; os 16 módulos solicitados foram verificados e o YAML central foi carregado para confirmar a seção `legacy.scale`. Também foi verificada estaticamente a ordem dimensional que causava o erro no Teste 13.

Não foi possível executar os testes funcionais nesta máquina porque PyTorch não está instalado (`ModuleNotFoundError: No module named 'torch'`). Os benchmarks 05, 12, 13 e 16 não foram executados aqui, por demanda de memória/tempo. Os resultados anexados são resultados da versão do usuário, não uma execução das alterações entregues.

## Como testar no ambiente NeuroCore

A partir da raiz do repositório, com as dependências do projeto instaladas, execute os comandos desejados, por exemplo:

```bash
python -m experiments.test_01_lif_base
python -m experiments.test_02_stdp
python -m experiments.test_03_rstdp
python -m experiments.test_04_neurocore_02
python -m experiments.test_05_scale_10k
# ... até test_16_dynamic_1m
```

Os testes de escala 10k/100k/1M podem consumir muita memória. Execute-os separadamente e monitore RAM/VRAM, especialmente `test_05_scale_10k`, `test_12_scale_100k`, `test_13_scale_1m` e `test_16_dynamic_1m`.
