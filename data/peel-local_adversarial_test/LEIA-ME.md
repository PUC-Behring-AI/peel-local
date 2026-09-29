# Auditoria do revisor adversarial (Fase 2) — PEEL-Local

Comece por **`RESULTS.md`**, que resume todos os resultados, com protocolo, tabelas e limitações da medição. (English translation: [`README.md`](README.md) / [`RESULTS.en.md`](RESULTS.en.md).)

| Caminho | Conteúdo |
|---|---|
| `RESULTS.md` | resumo dos resultados (todos os números vêm das tabelas abaixo) |
| `results/aws/summary_*.csv` | tabelas por configuração: grupos, categorias, similaridade, comprimento, integridade, máquina |
| `results/aws/fixed_input_aws.jsonl` | as 1.155 chamadas do revisor (385 por configuração), com a resposta bruta do modelo |
| `results/aws/env/` | ambiente de cada job: versão do Ollama, digests dos modelos, GPU |
| `results/mac_pilot/` | piloto no Mac (2 chamadas por configuração), só para comparação |
| `code/adversarial_review_audit/` | código da auditoria (`audit.py`, `classify.py`, `analyze.py`, testes, README) e jobs do SageMaker |
| `code/build_summary.py` | gera o `RESULTS.md` a partir das tabelas (`code/build_summary_en.py` gera a versão em inglês, `RESULTS.en.md`) |

**Direitos do texto-fonte.** O `.jsonl` contém as condensações revisadas e as respostas do modelo, que são derivadas de Boisseau (2026), sob CC BY-NC-ND 4.0. A permissão de redistribuição ainda está pendente: ver `PROVENANCE_source_Boisseau.md`.
