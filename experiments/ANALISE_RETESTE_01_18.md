# Análise dos testes refeitos — Testes 01–18

## Resumo

No log anexado, os 18 comandos terminaram sem traceback. Isso confirma que os erros de compatibilidade vistos anteriormente foram superados na cópia testada. Porém, conclusão sem exceção não significa que o agente aprendeu ou que todas as métricas demonstram o comportamento esperado.

## Aprendizagem e comportamento

- **Teste 01:** o teste unitário LIF passou. A simulação base registra 1 meta em 50 episódios, mas o próprio script não possui plasticidade; portanto, esse resultado é somente uma referência de navegação sem aprendizagem.
- **Teste 02 (STDP):** 1 meta em 50 episódios. Esse valor é igual ao observado no Teste 01, mas os ensaios não foram controlados por semente comum e isso não prova equivalência nem ganho de aprendizagem.
- **Teste 03 (R-STDP):** 3 metas em 50 episódios. É uma diferença pequena, sem múltiplas sementes ou controle estatístico; não basta para concluir que R-STDP melhorou a política.
- **Teste 04 (NeuroCore 0.2):** 0 metas em 30 episódios, embora o raster tenha sido salvo. A integração executou, mas o desempenho de navegação não demonstrou sucesso.
- **Teste 06:** 2 metas em 10 episódios (20%); resultado funcional, mas ainda baixo e baseado em poucos episódios.
- **Teste 07:** os episódios terminaram, e o checkpoint foi salvo. O log não exibe recompensas, contagem de metas nem diversidade de ações, então não permite avaliar aprendizagem.
- **Teste 08:** escolheu ação 0 nos 20 passos e recebeu recompensa -0,01 em todos. Isso é evidência de política colapsada ou circuito de decisão silencioso, não de comportamento cortical diverso. O script foi ajustado após esse log para amplificar a entrada atencional e imprimir a contagem de spikes e ações; essa nova alteração ainda precisa ser reexecutada.

## Memória, percepção e monitoramento

- **Teste 09:** recuperou exatamente o padrão em 5/5 ciclos. O teste usa uma matriz Hebbiana construída diretamente para o padrão; valida essa demonstração recorrente específica, não a classe/módulo de memória associativa de produção.
- **Teste 10:** os codificadores e integração LIF concluíram sem erro, conforme o log.
- **Teste 11:** a taxa de acerto exibida foi 80%; o custo energético é uma estimativa. O desvio homeostático 6,16 merece interpretação contra a meta configurada, não deve ser lido isoladamente como “passou”.
- **Teste 14:** os três gráficos foram gerados.
- **Teste 15:** fração média de atividade oculta foi 0,078291 no domínio A e 0,102748 no B, diferença absoluta 0,024456. Isso descreve mudança da atividade, não mede sucesso em tarefa ou generalização comportamental.

## Escala e homeostase

- **Teste 05:** 10k neurônios, construção em 10,84 s e simulação de 50 passos em 18,41 s. O aviso do PyTorch sobre invariantes esparsas não interrompeu a execução, mas é aviso de validação/performance, não prova de que todos os tensores esparsos foram verificados.
- **Teste 12:** 100k neurônios concluído; alocação em 1,73 s, ciclo em 226,61 ms e custo estimado de 0,00001562 J.
- **Teste 13:** 1M concluído; alocação em 25,91 s, ciclo em 2,28 s e custo estimado de 0,00010566 J. A versão atual corrige a contagem da arquitetura para totalizar exatamente 1M.
- **Teste 16:** 1.000 passos em 1.018,92 s (cerca de 17 min), média de 53.488 spikes por passo. NVML não inicializou; logo, não houve medição de energia nesse teste.
- **Teste 17:** os checks reportaram PASS, mas a atividade após retirar o estímulo foi zero, e o limiar após recuperação (1,90) permaneceu acima do valor anterior ao estímulo (0,50). O limiar caiu do pico de 3,26, mas esse resultado mostra estabilidade em silêncio, não retorno demonstrado a uma faixa ativa de disparo.
- **Teste 18:** todos os checks passaram: associação aprendida, retida sem estímulo e restaurada do snapshot; uma experiência episódica foi transferida e restaurada.

## Conclusão e próximo passo

Os erros de API e de chave de configuração foram resolvidos na cópia atual. O principal alerta comportamental deste novo log é o Teste 08, já instrumentado para diagnosticar silêncio/colapso de ação na próxima execução. Os resultados de aprendizagem (Testes 02–04 e 06–08) devem ser repetidos com sementes controladas e mais episódios antes de comparar regras de plasticidade. O Teste 16 é demorado e não mediu energia neste hardware.
