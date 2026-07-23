#!/usr/bin/env python3
"""
Avaliação de Retrieval do agente GraphRAG — TCC.

Mede o desempenho do agente ao recuperar informações dos grafos gerados
em cada cenário (Gabarito, A, B, C), comparando:
  - Velocidade (tempo de resposta)
  - Quantidade de nós acessados e seus vizinhos semânticos
  - Corretude/Precisão da resposta (LLM como juiz — score 0.0 a 1.0)

A resposta do Gabarito serve como referência para o juiz avaliar os demais.

CACHE DSPy: desativado por padrão para garantir chamadas reais ao LLM.

Uso (a partir de Agent_Rag/):
    python3 -m src.scripts.evaluate_retrieval \\
        --gabarito-gid "gabarito_tarefas" \\
        --scenario-gids A=eval_a_tarefas_... B=eval_b_tarefas_... C=eval_c_tarefas_...

    # Listar grafos disponíveis no Neo4j:
    python3 -m src.scripts.evaluate_retrieval --list-graphs
"""

import sys
import os
import argparse
import json
import re
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import dspy

from src.infra.neo4j.client import Neo4jClient
from src.infra.dspy.agent import build_agent
from src.infra.dspy.signatures import RequirementsQA
from src import config

# ─── Cores ────────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
DIM     = "\033[2m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


# ─── Perguntas de teste (temáticas) ──────────────────────────────────────────

DEFAULT_QUESTIONS = [
    "Quais são as principais funcionalidades para gerenciamento de tarefas (criação, edição, exclusão)?",
    "Quais são as dependências funcionais e estruturais para que a edição de tarefas funcione corretamente?",
]


# ─── Juiz LLM ─────────────────────────────────────────────────────────────────

_JUDGE_INSTRUCTIONS = """\
Você é um avaliador especialista em Engenharia de Requisitos.

Sua tarefa é avaliar a qualidade de uma RESPOSTA gerada por um agente de IA,
comparando-a com uma RESPOSTA DE REFERÊNCIA (obtida de um Gabarito base).

Critérios de avaliação:
1. CORRETUDE: A resposta aborda o que foi perguntado de forma correta e embasada nos nós do grafo?
2. COMPLETUDE: A resposta abrange os aspectos principais da referência?
3. COERÊNCIA: A resposta faz sentido para um engenheiro de software?
4. PROFUNDIDADE (GRAPH-RAG): A resposta demonstra conhecimento explícito de relações semânticas (depende de, estende, conflita com, impacta)?

REGRA DE OURO PARA PROFUNDIDADE:
O Gabarito de referência pode ser POBRE em relacionamentos semânticos (ele pode ter apenas agrupamentos simples).
Se a RESPOSTA DO CANDIDATO fornecer mapeamentos de dependências, extensões ou análises de impacto MAIS RICOS E PROFUNDOS que a própria referência, você DEVE dar NOTA MÁXIMA (1.0). Não penalize o candidato por ser melhor e mais estruturado que a referência.

Retorne SOMENTE um número de 0.0 a 1.0 representando a qualidade geral:
  1.0 = Equivalente ou SUPERIOR à referência em profundidade relacional.
  0.7 = Boa — responde à pergunta de forma competente.
  0.4 = Parcial — responde o básico, sem conexões claras ou faltando informações chave.
  0.1 = Fraca — não encontrou requisitos ou deu resposta superficial.
  0.0 = Inválida — erro ou alucinação total.

Não adicione texto, apenas o número (ex: 0.75).
"""

JudgeSignature = dspy.Signature(
    "question, reference_answer, candidate_answer -> score",
).with_instructions(_JUDGE_INSTRUCTIONS)


def _judge_answer(
    question: str,
    reference: str,
    candidate: str,
) -> float:
    """Usa o LLM como juiz para pontuar a resposta do candidato vs. referência.
    Cria um LM temporário sem cache para garantir avaliação fresca.
    """
    # LM sem cache especificamente para o juiz
    judge_lm = dspy.LM(
        model=config.DSPY_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        temperature=0.0,
        max_tokens=128,
        cache=False,
    )
    judge = dspy.Predict(JudgeSignature)
    try:
        with dspy.context(lm=judge_lm):
            res = judge(
                question=question,
                reference_answer=reference[:1200],
                candidate_answer=candidate[:1200],
            )
        raw = str(getattr(res, "score", "0.0")).strip()
        m = re.search(r"\d+\.\d+|\d+", raw)
        score = float(m.group(0)) if m else 0.0
        return min(max(round(score, 3), 0.0), 1.0)
    except Exception as e:
        print(f"    {_c(YELLOW, f'[juiz] erro ao pontuar: {e}')}")
        return 0.0


# ─── Enriquecimento de nós acessados ─────────────────────────────────────────

def _enrich_nodes(client: Neo4jClient, node_ids: List[str], graph_id: str) -> List[Dict]:
    """Busca o texto completo, metadados e vizinhos dos nós acessados no Neo4j."""
    if not node_ids:
        return []
    enriched = []
    for req_id in node_ids:
        ctx = client.get_requirement_context(req_id, graph_id=graph_id)
        if ctx:
            enriched.append({
                "req_id": ctx["req_id"],
                "text": ctx.get("text", ""),
                "summary": ctx.get("summary", ""),
                "type": ctx.get("type", ""),
                "domain": ctx.get("domain", ""),
                "communityId": ctx.get("communityId"),
                "techniques": ctx.get("techniques", []),
                "concepts": ctx.get("concepts", []),
                "neighbors": [
                    {
                        "req_id": nb.get("req_id"),
                        "text": nb.get("text", "")[:120],
                        "rel_type": nb.get("rel_type"),
                    }
                    for nb in (ctx.get("neighbors") or [])
                    if nb and nb.get("req_id")
                ],
            })
        else:
            enriched.append({"req_id": req_id, "text": None, "error": "não encontrado"})
    return enriched


# ─── Retrieval de um único grafo ──────────────────────────────────────────────

def _retrieve_one(
    agent,
    client: Neo4jClient,
    question: str,
    graph_id: str,
) -> Tuple[str, List[str], List[Dict], float]:
    """
    Executa o agente no graph_id e retorna
    (resposta, node_ids, nodes_enriched, tempo_s).
    Cache DSPy é desabilitado via variável de ambiente antes da chamada.
    """
    # Desabilita cache DSPy por chamada via env var
    os.environ["DSP_CACHEBOOL"] = "false"

    t0 = time.time()
    try:
        answer, node_ids = agent.ask(question=question, graph_id=graph_id)
    except Exception as e:
        answer = f"[ERRO: {e}]"
        node_ids = []
    elapsed = round(time.time() - t0, 3)

    # Enriquece nós com dados completos do Neo4j
    nodes_enriched = _enrich_nodes(client, node_ids, graph_id)

    return answer, node_ids, nodes_enriched, elapsed


# ─── Listar grafos disponíveis ────────────────────────────────────────────────

def _list_graphs(client: Neo4jClient) -> None:
    rows = client.run(
        "MATCH (g:Graph) RETURN g.graph_id AS gid, g.name AS name ORDER BY g.graph_id"
    )
    if not rows:
        rows = client.run(
            "MATCH (r:Requirement) RETURN DISTINCT r.graph_id AS gid, '' AS name ORDER BY r.graph_id"
        )
    print()
    print(_c(BOLD, "Grafos disponíveis no Neo4j:"))
    print(_c(DIM, "─" * 60))
    for r in rows:
        gid = r.get("gid", "?")
        name = r.get("name") or ""
        print(f"  {_c(CYAN, gid)}" + (f"  ({name})" if name else ""))
    print()


# ─── Avaliação completa ───────────────────────────────────────────────────────

def _run_evaluation(
    agent,
    client: Neo4jClient,
    questions: List[str],
    gabarito_gid: str,
    scenario_gids: Dict[str, str],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Para cada pergunta:
      1. Executa retrieval real (sem cache) no Gabarito (referência)
      2. Executa retrieval real (sem cache) em cada cenário (A, B, C)
      3. Juiz pontua cada cenário vs. referência
      4. Salva os nós completos (texto, metadados, vizinhos) no resultado
    """
    all_results: List[Dict[str, Any]] = []
    total_q = len(questions)

    for q_idx, question in enumerate(questions):
        if verbose:
            print()
            print(_c(BOLD + CYAN, f"  ┌─ [{q_idx+1}/{total_q}] {question[:80]}"))

        # ── Referência (Gabarito) ──────────────────────────────────────────────
        if verbose:
            print(_c(DIM, f"    Gabarito ({gabarito_gid}) ..."))
        ref_answer, ref_ids, ref_nodes, ref_time = _retrieve_one(agent, client, question, gabarito_gid)
        if verbose:
            print(_c(BLUE, f"    ⏱ {ref_time}s | {len(ref_ids)} nós acessados"))

        q_result: Dict[str, Any] = {
            "question": question,
            "gabarito": {
                "graph_id": gabarito_gid,
                "answer": ref_answer,
                "nodes_accessed_ids": ref_ids,
                "nodes_accessed_full": ref_nodes,   # ← nós completos com texto e vizinhos
                "num_nodes": len(ref_ids),
                "time_s": ref_time,
                "score": 1.0,
            },
            "scenarios": {},
        }

        # ── Cenários (A, B, C) ────────────────────────────────────────────────
        for sc_label, gid in scenario_gids.items():
            if verbose:
                print(_c(DIM, f"    Cenário {sc_label} ({gid}) ..."))
            ans, node_ids, nodes_enriched, elapsed = _retrieve_one(agent, client, question, gid)

            if verbose:
                print(_c(DIM, f"    Juiz avaliando Cenário {sc_label} ..."))
            score = _judge_answer(question, ref_answer, ans)

            color = RED if sc_label == "A" else (YELLOW if sc_label == "B" else GREEN)
            if verbose:
                print(_c(color, f"    [{sc_label}] ⏱ {elapsed}s | {len(node_ids)} nós | Score: {score:.2f}"))

            q_result["scenarios"][sc_label] = {
                "graph_id": gid,
                "answer": ans,
                "nodes_accessed_ids": node_ids,
                "nodes_accessed_full": nodes_enriched,  # ← nós completos
                "num_nodes": len(node_ids),
                "time_s": elapsed,
                "score": score,
            }

        all_results.append(q_result)

    return all_results


# ─── Tabela de resultados ─────────────────────────────────────────────────────

def _print_retrieval_summary(results: List[Dict[str, Any]]) -> None:
    all_scenarios: List[str] = []
    if results:
        all_scenarios = sorted(results[0].get("scenarios", {}).keys())

    print()
    print(_c(BOLD + MAGENTA, "═══ Resumo de Retrieval — Avaliação GraphRAG ═══"))
    print()

    gabarito_times: List[float] = []
    sc_times: Dict[str, List[float]] = {sc: [] for sc in all_scenarios}
    sc_scores: Dict[str, List[float]] = {sc: [] for sc in all_scenarios}
    sc_nodes: Dict[str, List[int]] = {sc: [] for sc in all_scenarios}
    gab_nodes: List[int] = []

    for r in results:
        gabarito_times.append(r["gabarito"]["time_s"])
        gab_nodes.append(r["gabarito"]["num_nodes"])
        for sc in all_scenarios:
            sc_data = r["scenarios"].get(sc, {})
            sc_times[sc].append(sc_data.get("time_s", 0.0))
            sc_scores[sc].append(sc_data.get("score", 0.0))
            sc_nodes[sc].append(sc_data.get("num_nodes", 0))

    def mean_std(vals: List[float]) -> str:
        if not vals:
            return "N/A"
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.2f} ±{s:.2f}"

    headers = ["Grafo/Cenário", "Tempo (s)", "Nós Acessados", "Score vs Gabarito"]
    col_w   = [40, 14, 16, 20]
    print(_c(BOLD, "  ".join(h.ljust(w) for h, w in zip(headers, col_w))))
    print(_c(DIM, "─" * 95))

    gab_t_str = mean_std(gabarito_times)
    gab_n_str = mean_std([float(n) for n in gab_nodes])
    print(_c(BLUE, "  ".join(v.ljust(w) for v, w in zip(
        ["Gabarito (Gold Standard)", gab_t_str, gab_n_str, "1.00 (referência)"],
        col_w
    ))))

    colors = {"A": RED, "B": YELLOW, "C": GREEN}
    sc_labels = {
        "A": "A — Modelo Cru (zero-shot)",
        "B": "B — Prompt Básico (RELATED_TO)",
        "C": "C — DSPy Otimizado",
    }
    for sc in all_scenarios:
        t_str = mean_std(sc_times[sc])
        n_str = mean_std([float(n) for n in sc_nodes[sc]])
        s_str = mean_std(sc_scores[sc])
        label = sc_labels.get(sc, f"Cenário {sc}")
        color = colors.get(sc, RESET)
        print(_c(color, "  ".join(v.ljust(w) for v, w in zip(
            [label, t_str, n_str, s_str],
            col_w
        ))))

    print()
    print(_c(BOLD, "  Ranking por Score médio:"))
    ranked = sorted(
        [(sc, statistics.mean(sc_scores[sc])) for sc in all_scenarios if sc_scores[sc]],
        key=lambda x: x[1], reverse=True
    )
    for pos, (sc, avg_score) in enumerate(ranked, 1):
        color = colors.get(sc, RESET)
        print(_c(color, f"    {pos}º {sc_labels.get(sc, sc)}: {avg_score:.3f}"))
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Avalia o retrieval do agente GraphRAG nos grafos de cada cenário.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gabarito-gid", type=str, default="",
        help="graph_id do Gabarito (Gold Standard). Ex: 'gabarito_tarefas'"
    )
    parser.add_argument(
        "--scenario-gids", type=str, nargs="*", default=[],
        metavar="LABEL=GRAPH_ID",
        help=(
            "graph_ids dos cenários no formato LABEL=graph_id.\n"
            "Exemplo: A=eval_a_tarefas_... B=eval_b_tarefas_... C=eval_c_tarefas_..."
        )
    )
    parser.add_argument("--list-graphs", action="store_true",
                        help="Lista grafos disponíveis no Neo4j e encerra.")
    parser.add_argument("--save-json", type=str, default="",
                        help="Caminho para salvar o JSON de resultados.")
    parser.add_argument("--questions-file", type=str, default="",
                        help="JSON com lista de perguntas personalizadas.")
    args = parser.parse_args()

    print()
    print(_c(BOLD + MAGENTA, "╔══════════════════════════════════════════════════════════════╗"))
    print(_c(BOLD + MAGENTA, "║         TCC — Avaliação de Retrieval GraphRAG                ║"))
    print(_c(BOLD + MAGENTA, "║  Gabarito vs A (cru) vs B (RELATED_TO) vs C (DSPy)           ║"))
    print(_c(BOLD + MAGENTA, "╚══════════════════════════════════════════════════════════════╝"))
    print()
    print(_c(YELLOW, "  ⚠ Cache DSPy DESABILITADO — todas as chamadas são reais ao LLM."))
    print()

    print(_c(DIM, "Conectando ao Neo4j..."))
    client = Neo4jClient()
    client.test_connection()
    print(_c(GREEN, "✓ Neo4j conectado"))

    if args.list_graphs:
        _list_graphs(client)
        client.close()
        return

    if not args.gabarito_gid:
        print(_c(RED, "✗ Informe --gabarito-gid. Use --list-graphs para ver os grafos disponíveis."))
        client.close()
        return

    scenario_gids: Dict[str, str] = {}
    for item in (args.scenario_gids or []):
        if "=" in item:
            label, gid = item.split("=", 1)
            scenario_gids[label.strip().upper()] = gid.strip()

    if not scenario_gids:
        print(_c(YELLOW, "⚠ Nenhum --scenario-gids fornecido. Avaliando apenas o Gabarito."))

    if args.questions_file and os.path.exists(args.questions_file):
        with open(args.questions_file, encoding="utf-8") as f:
            questions = json.load(f)
        print(_c(GREEN, f"✓ {len(questions)} perguntas carregadas de {args.questions_file}"))
    else:
        questions = DEFAULT_QUESTIONS
        print(_c(DIM, f"  Usando {len(questions)} perguntas padrão (tema: tarefas/produtividade)"))

    print(f"  {_c(BOLD, 'Gabarito:')}     {args.gabarito_gid}")
    for sc, gid in scenario_gids.items():
        print(f"  {_c(BOLD, f'Cenário {sc}:')}  {gid}")
    print(f"  {_c(BOLD, 'Perguntas:')}    {len(questions)}")
    print()

    # Configura DSPy e agente com cache=False
    print(_c(DIM, "Configurando DSPy/LLM e agente (cache=False)..."))
    lm = dspy.LM(
        model=config.DSPY_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        temperature=0.3,
        max_tokens=config.LLM_MAX_TOKENS,
        cache=False,   # ← Cache desabilitado globalmente
    )
    dspy.configure(lm=lm)
    agent = build_agent(client, cache=False)
    print(_c(GREEN, f"✓ Agente pronto | LLM: {config.DSPY_MODEL} | cache=False"))
    print()

    print(_c(BOLD + CYAN, "── Iniciando avaliação de retrieval (sem cache)..."))
    results = _run_evaluation(
        agent=agent,
        client=client,
        questions=questions,
        gabarito_gid=args.gabarito_gid,
        scenario_gids=scenario_gids,
        verbose=True,
    )

    _print_retrieval_summary(results)

    # Salva JSON completo (com nós enriquecidos)
    output = {
        "meta": {
            "gabarito_gid": args.gabarito_gid,
            "scenario_gids": scenario_gids,
            "model": config.DSPY_MODEL,
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "questions": questions,
            "cache": False,
            "note": (
                "score=1.0 para Gabarito; LLM como juiz para A/B/C vs. Gabarito. "
                "nodes_accessed_full contém texto, metadados e vizinhos semânticos de cada nó."
            ),
        },
        "results": results,
    }

    if args.save_json:
        save_path = args.save_json
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    else:
        auto_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "results")
        os.makedirs(auto_dir, exist_ok=True)
        stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(auto_dir, f"retrieval_{stamp}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(_c(GREEN, f"✓ Resultados salvos em: {save_path}"))

    client.close()
    print(_c(BOLD + GREEN, "✓ Avaliação de retrieval concluída!\n"))


if __name__ == "__main__":
    main()
