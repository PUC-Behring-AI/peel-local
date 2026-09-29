# Auditoria do revisor adversarial (Fase 2) — resumo dos resultados

Documento de trabalho para decidir o texto da Limitations e do §4.3. Nenhum número aqui foi digitado à mão: todos saem de `results/aws/summary_*.csv` e `results/mac_pilot/summary_*.csv`, gerados por `code/adversarial_review_audit/analyze.py`.

## 1. Pergunta

Com que frequência o revisor adversarial (a) não entrega nada ao pesquisador ou (b) devolve o mesmo texto que recebeu, nas três configurações do artigo? O desenho mede o **revisor isolado**: cada chamada recebe exatamente a condensação que a execução do artigo enviou a ele (lida do log de decisões), e só a amostragem do modelo varia.

## 2. Protocolo

| Item | Valor |
|---|---|
| Chamadas analisadas (AWS) | 1155 (qwen2b · 20 sent.: 385, qwen9b · 10 sent.: 385, qwen9b · 20 sent.: 385) |
| Modelos | `qwen3.5:2b` digest `324d162be6ca`, `qwen3.5:9b` digest `6488c96fa5fa` — verificados em todos os jobs |
| Ollama | 0.24.0 |
| GPUs | NVIDIA A10G, 23028 MiB, 595.91.07; NVIDIA L4, 23034 MiB, 595.91.07; Tesla T4, 15360 MiB, 595.91.07 |
| Contexto | `OLLAMA_CONTEXT_LENGTH=32768` (ver §7) |
| Chamada | a do pipeline: `run_adversarial_review` real, sem `temperature`/`seed`, `think=False`, timeout 600 s |
| Amostragem | parâmetros do Modelfile: temperatura 1, top_k 20, top_p 0,95, presence_penalty 1,5 |
| Intervalos | Wilson 95%, por célula |

**Identidade dos pesos.** Os modelos foram baixados em 28/09/2026, no dia do piloto. As execuções do artigo (09/09/2026) não registraram digest, então **não é possível afirmar que os pesos medidos aqui são os mesmos do artigo**. As taxas abaixo são dos digests acima.

## 3. Resultado principal

| Configuração | n | sem saída | mesmo texto (exato) | expandido | OK | alterado |
|---|---:|---|---|---|---|---|
| qwen2b · 20 sent. | 385 | 76,9% (72,4%–80,8%) | 0,0% (0,0%–1,0%) | 3,1% (1,8%–5,4%) | 17,9% (14,4%–22,1%) | 2,1% (1,1%–4,0%) |
| qwen9b · 10 sent. | 385 | 1,8% (0,9%–3,7%) | 7,0% (4,9%–10,0%) | 1,0% (0,4%–2,6%) | 61,3% (56,3%–66,0%) | 28,8% (24,5%–33,5%) |
| qwen9b · 20 sent. | 385 | 0,0% (0,0%–1,0%) | 10,1% (7,5%–13,5%) | 1,0% (0,4%–2,6%) | 27,8% (23,6%–32,5%) | 61,0% (56,1%–65,8%) |

### 3.1 Como ler esta tabela

Cada **linha** é uma configuração do artigo; as cinco **colunas** repartem as chamadas daquela configuração por desfecho, então cada linha soma 100% (a menos de arredondamento).

**As linhas**

| Linha | Quem gerou e quem revisou | O texto revisado |
|---|---|---|
| qwen2b · 20 sent. | `qwen3.5:2b` nos dois papéis | a condensação aceita na execução do artigo com 20 sentenças informativas por cluster: 747 palavras (alvo 3.456) |
| qwen9b · 10 sent. | `qwen3.5:9b` nos dois papéis | a condensação aceita na execução do artigo com 10 sentenças informativas por cluster: 3.062 palavras (alvo 3.456) |
| qwen9b · 20 sent. | `qwen3.5:9b` nos dois papéis | a condensação aceita na execução do artigo com 20 sentenças informativas por cluster: 3.079 palavras (alvo 3.456) |

"Sent." é o número de sentenças informativas por cluster dado ao modelo no prompt. Em cada linha o revisor recebeu **o mesmo texto 385 vezes**; só a amostragem do modelo (temperatura 1) varia de uma chamada para outra.

**As colunas**

O revisor é instruído a responder `VERDICT: OK` se não houver nenhum de três defeitos (fim abrupto, texto truncado ou repetido, conteúdo fora das sentenças informativas), ou `VERDICT: FIXED` seguido do texto com a menor correção possível.

| Coluna | O que o revisor fez | O que o pipeline faz | O que o log registra e o pesquisador vê |
|---|---|---|---|
| **sem saída** | Não entregou uma revisão utilizável: a chamada estourou o tempo limite, a resposta não tinha `VERDICT`, veio FIXED sem o texto corrigido, ou veio FIXED com um texto que o pipeline descartou pelo limiar de tamanho (L2: mínimo de 0,5 × alvo = 1.728 palavras) | Mantém o texto original | `unchanged` — indistinguível de "o revisor não achou nada", embora ele não tenha dito OK |
| **mesmo texto (exato)** | Declarou FIXED e devolveu o texto que recebeu, idêntico palavra por palavra (ignorando espaços) | Aceita como correção | **`fixed`** — falso sucesso: o relatório diz que houve correção, nada mudou |
| **expandido** | Declarou FIXED e devolveu um texto mais longo que 1,05 × o maior entre o texto revisado e o alvo; tipicamente o original com um bloco inteiro de texto novo anexado | Aceita — o pipeline só tem limite *inferior* de tamanho | **`fixed`** — a condensação final cresce e ganha material novo, o oposto da "menor correção possível" |
| **OK** | Declarou `VERDICT: OK` (sem defeito) | Mantém o texto original | `unchanged` — o funcionamento previsto; **não se verificou se o OK estava certo** |
| **alterado** | Declarou FIXED e devolveu um texto diferente, dentro do limite de tamanho | Aceita e substitui | **`fixed`** — o funcionamento previsto; **não se verificou se a edição corrigiu algo** |

Uma analogia para as duas primeiras colunas: o revisor é um revisor de provas que devolve o manuscrito ao editor. Em *sem saída*, ou ele não devolve nada, ou devolve correções que a expedição joga fora porque o envelope veio fino demais. Em *mesmo texto*, ele devolve o manuscrito carimbado "corrigido" sem ter mudado uma vírgula.

**Os números**

- O **percentual** é a fração das chamadas da linha que caiu naquela coluna.
- O **intervalo entre parênteses** é o intervalo de confiança de 95% (Wilson): a faixa em que a taxa verdadeira provavelmente está, dada a amostra. `0,0% (0,0%–1,0%)` quer dizer que nenhuma das 385 chamadas caiu ali e que a taxa real deve estar abaixo de 1%.
- **"Alterado" inclui textos quase idênticos.** A coluna *mesmo texto* conta só igualdade exata; uma "correção" que só acrescenta um ponto final ou troca uma palavra cai em *alterado*. qwen9b · 10 sent.: 18,7% das chamadas devolveram um texto aceito com similaridade ≥ 0,99; descontados os 7,0% exatos, 11,7% estão dentro de *alterado* mas são praticamente o mesmo texto; qwen9b · 20 sent.: 32,2% das chamadas devolveram um texto aceito com similaridade ≥ 0,99; descontados os 10,1% exatos, 22,1% estão dentro de *alterado* mas são praticamente o mesmo texto. A §5 mostra essa parcela em cada limiar.

	**Por que o 2b tem 76,9% de "sem saída".** É sobretudo um defeito do código, não do modelo. O texto revisado tem 747 palavras, e a regra L2 só aceita uma correção com pelo menos 1.728 palavras, então qualquer correção fiel desse texto é descartada. Isso aconteceu em 58,2% das chamadas. O restante vem do modelo ou da chamada: 13,5% FIXED sem o texto corrigido, 3,6% tempo limite de 600 s estourado, 1,6% sem `VERDICT`.

## 4. Decomposição de "sem saída"

| Configuração | falha/timeout da chamada | resposta vazia | sem `VERDICT` | FIXED sem `TEXT` | FIXED rejeitado pela L2 |
|---|---|---|---|---|---|
| qwen2b · 20 sent. | 14 · 3,6% (2,2%–6,0%) | 0 · 0,0% (0,0%–1,0%) | 6 · 1,6% (0,7%–3,4%) | 52 · 13,5% (10,5%–17,3%) | 224 · 58,2% (53,2%–63,0%) |
| qwen9b · 10 sent. | 1 · 0,3% (0,0%–1,5%) | 0 · 0,0% (0,0%–1,0%) | 0 · 0,0% (0,0%–1,0%) | 0 · 0,0% (0,0%–1,0%) | 6 · 1,6% (0,7%–3,4%) |
| qwen9b · 20 sent. | 0 · 0,0% (0,0%–1,0%) | 0 · 0,0% (0,0%–1,0%) | 0 · 0,0% (0,0%–1,0%) | 0 · 0,0% (0,0%–1,0%) | 0 · 0,0% (0,0%–1,0%) |

`FIXED rejeitado pela L2` é defeito do **código** (o modelo produziu texto; o pipeline o descartou pelo limiar de 0,5 × alvo). As outras quatro são do **modelo** ou da chamada. `falha/timeout` depende do hardware (ver §7).

Todas as falhas de chamada foram **timeouts de 600 s** (o limite do próprio pipeline), na execução principal em A10G, com o 2b dividindo a GPU com o 9b. A parcela tende a depender da velocidade da máquina (não testado separadamente).

## 5. "Mesmo texto" em todas as leituras

Parcela das chamadas cuja correção **aceita** tem similaridade de palavras com o texto revisado ≥ limiar (inclui as idênticas).

| Configuração | exato | ≥ 0,999 | ≥ 0,99 | ≥ 0,95 |
|---|---|---|---|---|
| qwen2b · 20 sent. | 0,0% (0,0%–1,0%) | 0,0% (0,0%–1,0%) | 0,0% (0,0%–1,0%) | 0,0% (0,0%–1,0%) |
| qwen9b · 10 sent. | 7,0% (4,9%–10,0%) | 8,8% (6,4%–12,1%) | 18,7% (15,1%–22,9%) | 30,4% (26,0%–35,2%) |
| qwen9b · 20 sent. | 10,1% (7,5%–13,5%) | 14,8% (11,6%–18,7%) | 32,2% (27,7%–37,0%) | 51,2% (46,2%–56,1%) |

## 6. Comprimento das correções devolvidas

Razão palavras da correção / palavras do texto revisado, para toda resposta FIXED com texto (aceita ou rejeitada).

| Configuração | n | mín | p05 | mediana | p95 | máx | > 1,05 | < 0,95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2b · 20 sent. | 244 | 0,03 | 0,91 | 0,98 | 4,13 | 14,63 | 13,9% | 15,6% |
| qwen9b · 10 sent. | 148 | 0,37 | 0,63 | 0,99 | 1,05 | 2,03 | 5,4% | 18,2% |
| qwen9b · 20 sent. | 278 | 0,65 | 0,76 | 0,99 | 1,02 | 1,77 | 2,5% | 28,4% |

## 7. Integridade e robustez

- qwen2b · 20 sent.: maior contexto usado 24194 tokens (limite 32.768); 0 chamadas no limite; 0 com `done_reason` ≠ `stop`.
- qwen9b · 10 sent.: maior contexto usado 13716 tokens (limite 32.768); 0 chamadas no limite; 0 com `done_reason` ≠ `stop`.
- qwen9b · 20 sent.: maior contexto usado 23059 tokens (limite 32.768); 0 chamadas no limite; 0 com `done_reason` ≠ `stop`.
- Classificação × decisão registrada pelo pipeline: `analyze.py` sai com erro em qualquer divergência; saiu sem erro.
- Mesma GPU (A10G), máquinas diferentes na execução principal — qui-quadrado de homogeneidade dos grupos:
  - qwen2b · 20 sent.: χ² = 3,70, gl = 3, p = 0,295
  - qwen9b · 10 sent.: χ² = 6,20, gl = 4, p = 0,185
  - qwen9b · 20 sent.: χ² = 2,85, gl = 3, p = 0,415
- Todas as máquinas (piloto em 6 tipos + execução principal):
  - qwen2b · 20 sent.: 8 máquinas, n = 385, χ² = 45,92, p de permutação = 0,060 (10.000 permutações; o piloto tem n = 2 por máquina, então o teste tem pouco poder para essas máquinas)
  - qwen9b · 10 sent.: 8 máquinas, n = 385, χ² = 18,86, p de permutação = 0,523 (10.000 permutações; o piloto tem n = 2 por máquina, então o teste tem pouco poder para essas máquinas)
  - qwen9b · 20 sent.: 8 máquinas, n = 385, χ² = 10,08, p de permutação = 0,860 (10.000 permutações; o piloto tem n = 2 por máquina, então o teste tem pouco poder para essas máquinas)
- Piloto no Mac (Apple Metal, contexto 131.072, n = 2 por configuração), só para comparação qualitativa:
  - qwen2b · 20 sent.: sem saída 2
  - qwen9b · 10 sent.: OK 1, alterado 1
  - qwen9b · 20 sent.: expandido 1, OK 1

## 8. O que esta medição não cobre

- **Três textos de entrada.** O desenho de entrada fixa mede o revisor nas três condensações do artigo; a taxa em outros textos não foi estimada. O teste de pipeline completo (30 réplicas) não coube no orçamento de US$ 30 e não foi executado.
- **Correção semântica.** `OK` e `alterado` são classificações de forma. Não se avaliou se um OK estava certo nem se uma alteração corrigiu algo.
- **Pesos do artigo.** Ver §2.
- **Motor.** CUDA (AWS) em vez do Metal do piloto no Mac; pesos idênticos aos do piloto. A máquina em que as execuções do artigo rodaram não está registrada.

## 9. Custo e tempo

| Etapa | Instância | Status | Faturado (s) | US$ |
|---|---|---|---:|---:|
| pilot | ml.g6.xlarge | Completed | 655 | 0,35 |
| pilot | ml.g4dn.xlarge | Completed | 716 | 0,25 |
| pilot | ml.g4dn.2xlarge | Completed | 641 | 0,28 |
| pilot | ml.g5.xlarge | Completed | 701 | 0,42 |
| pilot | ml.g5.2xlarge | Completed | 591 | 0,42 |
| pilot | ml.g6.2xlarge | Completed | 560 | 0,32 |
| main | ml.g5.xlarge | Completed | 17858 | 10,60 |
| main | ml.g5.2xlarge | Completed | 17930 | 12,83 |
| **total** | | | **39652** | **25,47** |

Preços: lista pública da AWS, SageMaker Training, sa-east-1.

## 10. Arquivos

- `results/aws/fixed_input_aws.jsonl` — todas as chamadas AWS, com a resposta bruta do modelo
- `results/aws/summary_*.csv` — tabelas de onde saem os números acima
- `results/mac_pilot/` — o piloto no Mac
- `results/aws/env/` — ambiente de cada job (versões, digests, GPU)
