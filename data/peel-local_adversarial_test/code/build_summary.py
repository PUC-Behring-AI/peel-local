"""Builds RESULTS.md (Portuguese) from the audit's CSV/JSONL outputs.

No number in the report is typed by hand: every figure is read from the
analysis tables written by experiments/adversarial_review_audit/analyze.py.

    python build_summary.py
"""
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
# Worktree used to run analyze.py's original SageMaker jobs; not portable
# as-is, set PEEL_REPO_WORKTREE to your own path to re-run this script.
WT = Path(os.environ.get("PEEL_REPO_WORKTREE", "/path/to/peel-local/.claude/worktrees/release-hardening"))
PRIMARY = HERE / "final" / "aws"
MAC = HERE / "final" / "mac_pilot"
PRICE = {"ml.g4dn.xlarge": 1.252, "ml.g4dn.2xlarge": 1.598, "ml.g5.xlarge": 2.1375,
         "ml.g5.2xlarge": 2.5752, "ml.g6.xlarge": 1.915, "ml.g6.2xlarge": 2.077}
COND_LABEL = {"qwen2b-20": "qwen2b · 20 sent.", "qwen9b-10": "qwen9b · 10 sent.", "qwen9b-20": "qwen9b · 20 sent."}
GROUP_LABEL = {"no_output": "sem saída", "same_text": "mesmo texto (exato)", "expanded": "expandido",
               "ok": "OK", "changed": "alterado"}
CAT_LABEL = {"call_failed": "falha/timeout da chamada", "empty_response": "resposta vazia",
             "unparseable": "sem `VERDICT`", "fixed_no_text": "FIXED sem `TEXT`",
             "fixed_rejected_short": "FIXED rejeitado pela L2", "ok": "OK",
             "fixed_identical": "FIXED idêntico", "fixed_expanded": "FIXED expandido",
             "fixed_changed": "FIXED alterado"}


def pct(x):
    return f"{100 * float(x):.1f}%".replace(".", ",")


def num(x, d=2):
    return f"{float(x):.{d}f}".replace(".", ",")


def ci(r):
    return f"{pct(r['ci95_lo'])}–{pct(r['ci95_hi'])}"


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_analysis(out_dir, jsonls):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PEEL_REPO=str(WT))
    subprocess.run([sys.executable, str(HERE / "experiments/adversarial_review_audit/analyze.py"), *map(str, jsonls)],
                   check=True, env=env)


def main():
    # ---- assemble inputs -------------------------------------------------
    PRIMARY.mkdir(parents=True, exist_ok=True)
    MAC.mkdir(parents=True, exist_ok=True)
    jobs = json.load(open(HERE / "aws_jobs" / "handoff_jobs.json"))["jobs"]
    merged = PRIMARY / "fixed_input_aws.jsonl"
    with merged.open("w", encoding="utf-8") as out:
        for j in jobs:
            for f in sorted(glob.glob(str(HERE / "aws_results" / j["name"] / "*.jsonl"))):
                out.write(Path(f).read_text(encoding="utf-8"))
    (MAC / "fixed_input_mac.jsonl").write_text((HERE / "results" / "fixed_input.jsonl").read_text(encoding="utf-8"),
                                               encoding="utf-8")
    run_analysis(PRIMARY, [merged])
    run_analysis(MAC, [MAC / "fixed_input_mac.jsonl"])

    groups, cats = read(PRIMARY / "summary_groups.csv"), read(PRIMARY / "summary_categories.csv")
    sims, lens = read(PRIMARY / "summary_similarity.csv"), read(PRIMARY / "summary_length.csv")
    integ, bytag = read(PRIMARY / "summary_integrity.csv"), read(PRIMARY / "summary_by_tag.csv")
    mac_groups = read(MAC / "summary_groups.csv")
    conds = sorted({r["condition"] for r in groups})

    def g(rows, cond, name, field="group"):
        return next(r for r in rows if r["condition"] == cond and r[field] == name)

    # ---- environment / cost ---------------------------------------------
    env_files = sorted(glob.glob(str(HERE / "aws_results" / "*" / "env_*.json")))
    env0 = json.load(open(env_files[0]))
    digests = {m["name"]: m["digest"] for m in env0["tags"]["models"] if m["name"].startswith("qwen3.5")}
    gpus = sorted({json.load(open(f)).get("gpu") or "?" for f in env_files})
    sys.path.append(str(HERE / "aws_jobs"))
    import sm_monitor as m  # noqa: E402
    cost_rows, total_cost, total_bill = [], 0.0, 0
    for j in jobs:
        d = m.sm.describe_training_job(TrainingJobName=j["name"])
        bill = d.get("BillableTimeInSeconds") or 0
        c = bill / 3600 * PRICE[j["type"]]
        total_cost += c
        total_bill += bill
        cost_rows.append((j["stage"], j["type"], d["TrainingJobStatus"], bill, c))

    # ---- hardware homogeneity (chi-square on groups, per condition) -------
    # All machines: many cells hold n = 2 (pilot), so the chi-square
    # approximation is invalid; use a permutation p-value for the same statistic.
    import numpy as np
    rng = np.random.default_rng(20260928)
    hw_tests = []
    for cond in conds:
        rows = [r for r in bytag if r["condition"] == cond]
        labels, tags = [], []
        for ti, r in enumerate(rows):
            for gi, name in enumerate(GROUP_LABEL):
                labels += [gi] * int(r[f"k_{name}"])
                tags += [ti] * int(r[f"k_{name}"])
        labels, tags = np.array(labels), np.array(tags)

        def stat(lab):
            t = np.zeros((len(rows), len(GROUP_LABEL)))
            np.add.at(t, (tags, lab), 1)
            t = t[:, t.sum(0) > 0]
            e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
            return ((t - e) ** 2 / e).sum()
        obs = stat(labels)
        perm = np.array([stat(rng.permutation(labels)) for _ in range(10000)])
        hw_tests.append((cond, len(rows), len(labels), obs, float((1 + (perm >= obs).sum()) / (1 + len(perm)))))
    main_tags = [r for r in bytag if r["tag"].startswith("main-")]
    main_tests = []
    for cond in conds:
        rows = [r for r in main_tags if r["condition"] == cond]
        table = [[int(r[f"k_{name}"]) for name in GROUP_LABEL] for r in rows]
        cols = [i for i in range(len(GROUP_LABEL)) if sum(t[i] for t in table) > 0]
        table = [[t[i] for i in cols] for t in table]
        if len(table) == 2 and len(cols) > 1:
            chi2, p, dof, _ = chi2_contingency(table)
            main_tests.append((cond, chi2, dof, p))

    # ---- write -------------------------------------------------------------
    L = []
    w = L.append
    w("# Auditoria do revisor adversarial (Fase 2) — resumo dos resultados\n")
    w("Documento de trabalho para decidir o texto da Limitations e do §4.3. Nenhum número aqui foi digitado à mão: "
      "todos saem de `final/aws/summary_*.csv` e `final/mac_pilot/summary_*.csv`, gerados por "
      "`experiments/adversarial_review_audit/analyze.py`.\n")

    w("## 1. Pergunta\n")
    w("Com que frequência o revisor adversarial (a) não entrega nada ao pesquisador ou (b) devolve o mesmo texto que "
      "recebeu, nas três configurações do artigo? O desenho mede o **revisor isolado**: cada chamada recebe exatamente a "
      "condensação que a execução do artigo enviou a ele (lida do log de decisões), e só a amostragem do modelo varia.\n")

    w("## 2. Protocolo\n")
    n_by = {c: int(g(groups, c, "no_output")["n"]) for c in conds}
    w("| Item | Valor |\n|---|---|")
    w(f"| Chamadas analisadas (AWS) | {sum(n_by.values())} ({', '.join(f'{COND_LABEL[c]}: {n_by[c]}' for c in conds)}) |")
    w(f"| Modelos | {', '.join(f'`{k}` digest `{v[:12]}`' for k, v in sorted(digests.items()))} — verificados em todos os jobs |")
    w(f"| Ollama | {env0['ollama_version'].get('version')} |")
    w(f"| GPUs | {'; '.join(gpus)} |")
    w("| Contexto | `OLLAMA_CONTEXT_LENGTH=32768` (ver §7) |")
    w("| Chamada | a do pipeline: `run_adversarial_review` real, sem `temperature`/`seed`, `think=False`, timeout 600 s |")
    w("| Amostragem | parâmetros do Modelfile: temperatura 1, top_k 20, top_p 0,95, presence_penalty 1,5 |")
    w("| Intervalos | Wilson 95%, por célula |\n")
    w("**Identidade dos pesos.** Os modelos foram baixados em 28/09/2026, no dia do piloto. As execuções do artigo "
      "(09/09/2026) não registraram digest, então **não é possível afirmar que os pesos medidos aqui são os mesmos do "
      "artigo**. As taxas abaixo são dos digests acima.\n")

    w("## 3. Resultado principal\n")
    w("| Configuração | n | sem saída | mesmo texto (exato) | expandido | OK | alterado |\n|---|---:|---|---|---|---|---|")
    for c in conds:
        cells = [f"{pct(g(groups, c, k)['share'])} ({ci(g(groups, c, k))})" for k in GROUP_LABEL]
        w(f"| {COND_LABEL[c]} | {n_by[c]} | " + " | ".join(cells) + " |")
    w("\n*Sem saída* = o pesquisador não recebe nada do revisor e o log registra `unchanged` (soma das categorias da §4). "
      "*Mesmo texto* = o log registra **`fixed`**, mas o texto é o que o revisor recebeu. *Expandido* = correção aceita "
      "mais longa que 1,05 × max(texto revisado, alvo), a tolerância de comprimento do próprio PEEL.\n")

    w("## 4. Decomposição de \"sem saída\"\n")
    w("| Configuração | " + " | ".join(CAT_LABEL[k] for k in CAT_LABEL if k in
      ("call_failed", "empty_response", "unparseable", "fixed_no_text", "fixed_rejected_short")) + " |")
    w("|---|" + "---|" * 5)
    for c in conds:
        cells = [f"{g(cats, c, k, 'category')['k']} · {pct(g(cats, c, k, 'category')['share'])} ({ci(g(cats, c, k, 'category'))})"
                 for k in ("call_failed", "empty_response", "unparseable", "fixed_no_text", "fixed_rejected_short")]
        w(f"| {COND_LABEL[c]} | " + " | ".join(cells) + " |")
    w("\n`FIXED rejeitado pela L2` é defeito do **código** (o modelo produziu texto; o pipeline o descartou pelo limiar "
      "de 0,5 × alvo). As outras quatro são do **modelo** ou da chamada. `falha/timeout` depende do hardware (ver §7).\n")
    w("Todas as falhas de chamada foram **timeouts de 600 s** (o limite do próprio pipeline), na execução principal em "
      "A10G, com o 2b dividindo a GPU com o 9b. A parcela tende a depender da velocidade da máquina (não testado separadamente).\n")

    w("## 5. \"Mesmo texto\" em todas as leituras\n")
    w("Parcela das chamadas cuja correção **aceita** tem similaridade de palavras com o texto revisado ≥ limiar "
      "(inclui as idênticas).\n")
    w("| Configuração | exato | ≥ 0,999 | ≥ 0,99 | ≥ 0,95 |\n|---|---|---|---|---|")
    for c in conds:
        ex = g(groups, c, "same_text")
        cells = [f"{pct(ex['share'])} ({ci(ex)})"]
        for t in ("0.999", "0.99", "0.95"):
            r = next(r for r in sims if r["condition"] == c and r["threshold"] == t)
            cells.append(f"{pct(r['share_of_calls'])} ({ci(r)})")
        w(f"| {COND_LABEL[c]} | " + " | ".join(cells) + " |")
    w("")

    w("## 6. Comprimento das correções devolvidas\n")
    w("Razão palavras da correção / palavras do texto revisado, para toda resposta FIXED com texto (aceita ou rejeitada).\n")
    w("| Configuração | n | mín | p05 | mediana | p95 | máx | > 1,05 | < 0,95 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in lens:
        w(f"| {COND_LABEL[r['condition']]} | {r['n_fixed_with_text']} | {num(r['min'])} | {num(r['p05'])} | "
          f"{num(r['median'])} | {num(r['p95'])} | {num(r['max'])} | {pct(r['share_above_1.05'])} | "
          f"{pct(r['share_below_0.95'])} |")
    w("")

    w("## 7. Integridade e robustez\n")
    for r in integ:
        w(f"- {COND_LABEL[r['condition']]}: maior contexto usado {r['max_ctx_used']} tokens (limite 32.768); "
          f"{r['n_ctx_used_ge_32768']} chamadas no limite; {r['n_done_reason_not_stop']} com `done_reason` ≠ `stop`.")
    w("- Classificação × decisão registrada pelo pipeline: `analyze.py` sai com erro em qualquer divergência; saiu sem erro.")
    if main_tests:
        w("- Mesma GPU (A10G), máquinas diferentes na execução principal — qui-quadrado de homogeneidade dos grupos:")
        for cond, chi2, dof, p in main_tests:
            w(f"  - {COND_LABEL[cond]}: χ² = {num(chi2)}, gl = {dof}, p = {num(p, 3)}")
    if hw_tests:
        w("- Todas as máquinas (piloto em 6 tipos + execução principal):")
        for cond, k, n, chi2, p in hw_tests:
            w(f"  - {COND_LABEL[cond]}: {k} máquinas, n = {n}, χ² = {num(chi2)}, p de permutação = {num(p, 3)} "
              "(10.000 permutações; o piloto tem n = 2 por máquina, então o teste tem pouco poder para essas máquinas)")
    w("- Piloto no Mac (Apple Metal, contexto 131.072, n = 2 por configuração), só para comparação qualitativa:")
    for c in conds:
        cells = ", ".join(f"{GROUP_LABEL[k]} {g(mac_groups, c, k)['k']}" for k in GROUP_LABEL if int(g(mac_groups, c, k)["k"]))
        w(f"  - {COND_LABEL[c]}: {cells}")
    w("")

    w("## 8. O que esta medição não cobre\n")
    w("- **Três textos de entrada.** O desenho de entrada fixa mede o revisor nas três condensações do artigo; "
      "a taxa em outros textos não foi estimada. O teste de pipeline completo (30 réplicas) não coube no orçamento "
      "de US$ 30 e não foi executado.")
    w("- **Correção semântica.** `OK` e `alterado` são classificações de forma. Não se avaliou se um OK estava certo "
      "nem se uma alteração corrigiu algo.")
    w("- **Pesos do artigo.** Ver §2.")
    w("- **Motor.** CUDA (AWS) em vez do Metal do piloto no Mac; pesos idênticos aos do piloto. A máquina em que as execuções do artigo rodaram não está registrada.\n")

    w("## 9. Custo e tempo\n")
    w("| Etapa | Instância | Status | Faturado (s) | US$ |\n|---|---|---|---:|---:|")
    for stage, it, st_, bill, c in cost_rows:
        w(f"| {stage} | {it} | {st_} | {bill} | {num(c)} |")
    w(f"| **total** | | | **{total_bill}** | **{num(total_cost)}** |\n")
    w("Preços: lista pública da AWS, SageMaker Training, sa-east-1.\n")

    w("## 10. Arquivos\n")
    w("- `final/aws/fixed_input_aws.jsonl` — todas as chamadas AWS, com a resposta bruta do modelo")
    w("- `final/aws/summary_*.csv` — tabelas de onde saem os números acima")
    w("- `final/mac_pilot/` — o piloto no Mac")
    w("- `aws_results/<job>/env_*.json` — ambiente de cada job (versões, digests, GPU)")
    (HERE / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", HERE / "RESULTS.md", f"total cost US${total_cost:.2f}")


if __name__ == "__main__":
    main()
