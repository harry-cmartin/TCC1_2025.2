#!/usr/bin/env python3
"""
Construção do Gabarito (Gold Standard) — TCC.

Lê user_stories_embeddings.csv, filtra um subconjunto temático,
carrega os nós no Neo4j, aplica similaridade de cosseno para criar
arestas SIMILAR_TO e roda Louvain para detectar comunidades.

Uso (a partir de Agent_Rag/):
    python -m src.scripts.build_gabarito --topic "tarefas" --tau 0.82 --limit 50
    python -m src.scripts.build_gabarito --topic "tarefas" --scan-tau  # escaneia limiares
    python -m src.scripts.build_gabarito --topic "tarefas" --cleanup   # remove gabarito existente e recria
"""

import sys
import os
import argparse
import ast
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.algorithms.community import louvain_communities

from src.infra.neo4j.client import Neo4jClient
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

def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


# ─── Keywords por tema ────────────────────────────────────────────────────────

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "tarefas": [
        "tarefa", "tarefas", "lista", "concluir", "prioridade", "prazo",
        "lembrete", "notificacao", "notificação", "calendario", "calendário",
        "evento", "agenda", "anotacao", "anotação", "organizar", "produtividade",
        "projeto", "categoria", "filtro", "pesquisar", "sincronizar",
    ],
    "restaurante": [
        "restaurante", "cardapio", "cardápio", "pedido", "mesa", "garcom",
        "garçom", "reserva", "prato", "cozinha", "cliente", "comida",
        "delivery", "refeicao", "refeição", "menu", "estoque", "pagamento",
    ],
    "banco": [
        "banco", "conta", "transferencia", "transferência", "saldo",
        "pagamento", "saque", "deposito", "depósito", "cartao", "cartão",
        "pix", "extrato", "agencia", "agência", "financeiro",
    ],
    "hospital": [
        "hospital", "paciente", "medico", "médico", "consulta", "exame",
        "prontuario", "prontuário", "leito", "enfermagem", "prescricao",
        "prescrição", "internacao", "internação", "laudo", "triagem",
    ],
}

def _get_keywords(topic: str) -> List[str]:
    t = topic.lower().strip()
    for key, kws in _TOPIC_KEYWORDS.items():
        if key in t or t in key:
            return kws
    # fallback: palavras do próprio topic
    return [w for w in t.split() if len(w) > 2] or [t]


# ─── Carga e filtragem do CSV ─────────────────────────────────────────────────

def _load_csv(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV de embeddings com separador ';'."""
    print(_c(DIM, f"  Lendo {csv_path} ..."))
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    print(_c(DIM, f"  {len(df)} user stories carregadas."))
    return df


def _parse_embedding(raw: str) -> List[float]:
    """Converte string de lista Python para lista de floats."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def _filter_by_topic(df: pd.DataFrame, topic: str, limit: int) -> pd.DataFrame:
    """Filtra user stories que mencionam keywords do tema."""
    keywords = _get_keywords(topic)
    mask = df["user_story"].str.lower().apply(
        lambda text: any(kw in text for kw in keywords)
    )
    filtered = df[mask].copy()

    if len(filtered) == 0:
        print(_c(YELLOW, f"  ⚠ Nenhuma user story encontrada para o tema '{topic}'. Usando amostra aleatória."))
        filtered = df.sample(min(limit, len(df)), random_state=42)
    elif len(filtered) > limit:
        filtered = filtered.sample(limit, random_state=42)

    print(_c(GREEN, f"  ✓ {len(filtered)} user stories selecionadas para o tema '{topic}'"))
    return filtered.reset_index(drop=True)


# ─── Persistência no Neo4j ────────────────────────────────────────────────────

def _gabarito_graph_id(topic: str) -> str:
    safe = re.sub(r"[^a-z0-9]", "_", topic.lower().strip())[:20]
    return f"gabarito_{safe}"


def _delete_gabarito(client: Neo4jClient, gid: str) -> None:
    print(_c(DIM, f"  Removendo gabarito existente: {gid} ..."))
    client.run("MATCH (r:Requirement {graph_id:$gid}) DETACH DELETE r", {"gid": gid})
    client.run("MATCH (g:Graph {graph_id:$gid}) DETACH DELETE g", {"gid": gid})
    print(_c(YELLOW, f"  ⚠ Gabarito '{gid}' removido."))


def _save_nodes(
    client: Neo4jClient,
    gid: str,
    topic: str,
    df: pd.DataFrame,
) -> Tuple[List[str], List[List[float]]]:
    """Persiste user stories como nós :Requirement no Neo4j. Retorna (req_ids, embeddings)."""
    client.create_graph_meta(gid, f"Gabarito — {topic}")

    req_ids: List[str] = []
    embeddings: List[List[float]] = []

    for i, row in df.iterrows():
        req_id = f"GAB_{re.sub(r'[^a-z0-9]', '_', topic.lower())[:6].upper()}_{i:04d}"
        text = str(row.get("user_story", "")).strip()
        summary = str(row.get("acceptance_criteria", "")).strip()
        emb = _parse_embedding(str(row.get("embedding", "[]")))

        client.create_requirement(
            req_id=req_id,
            text=text,
            summary=summary,
            req_type="funcional",
            domain=topic,
            source="gabarito_csv",
            embedding=emb,
            graph_id=gid,
        )
        req_ids.append(req_id)
        embeddings.append(emb)

    print(_c(GREEN, f"  ✓ {len(req_ids)} nós persistidos no Neo4j (graph_id={gid!r})"))
    return req_ids, embeddings


# ─── Arestas semânticas (Cosseno) ────────────────────────────────────────────

def _build_similarity_edges(
    client: Neo4jClient,
    gid: str,
    req_ids: List[str],
    embeddings: List[List[float]],
    tau: float,
) -> int:
    """Cria arestas SIMILAR_TO entre pares com cosine_similarity >= tau."""
    mat = np.array(embeddings, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2:
        print(_c(YELLOW, "  ⚠ Embeddings insuficientes para calcular similaridade."))
        return 0

    sim = cosine_similarity(mat)
    n = len(req_ids)
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= tau:
                client.run(
                    "MATCH (a:Requirement {req_id:$a, graph_id:$gid}), "
                    "(b:Requirement {req_id:$b, graph_id:$gid}) "
                    "MERGE (a)-[:SIMILAR_TO {score:$score, source:'cosine'}]->(b)",
                    {"a": req_ids[i], "b": req_ids[j], "gid": gid, "score": round(float(sim[i, j]), 4)},
                )
                count += 1

    print(_c(GREEN, f"  ✓ {count} arestas SIMILAR_TO criadas (tau={tau})"))
    return count


# ─── Louvain ─────────────────────────────────────────────────────────────────

def _run_louvain(
    client: Neo4jClient,
    gid: str,
    req_ids: List[str],
    embeddings: List[List[float]],
    tau: float,
) -> Tuple[int, float]:
    """Roda Louvain no grafo de similaridade e grava communityId nos nós."""
    mat = np.array(embeddings, dtype=float)
    sim = cosine_similarity(mat)
    n = len(req_ids)

    G = nx.Graph()
    G.add_nodes_from(req_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= tau:
                G.add_edge(req_ids[i], req_ids[j], weight=float(sim[i, j]))

    if G.number_of_edges() == 0:
        print(_c(YELLOW, "  ⚠ Nenhuma aresta para Louvain. Comunidades não detectadas."))
        return 0, 0.0

    communities = louvain_communities(G, seed=42)
    modularity = nx.community.modularity(G, communities)

    for comm_id, community in enumerate(communities):
        for req_id in community:
            client.run(
                "MATCH (r:Requirement {req_id:$req_id, graph_id:$gid}) SET r.communityId = $cid",
                {"req_id": req_id, "gid": gid, "cid": comm_id},
            )

    num_comms = len(communities)
    print(_c(GREEN, f"  ✓ Louvain: {num_comms} comunidades | Modularidade Q={modularity:.4f}"))
    return num_comms, modularity


# ─── Scan de tau ─────────────────────────────────────────────────────────────

def _scan_tau(embeddings: List[List[float]]) -> None:
    """Imprime tabela tau × arestas × comunidades × modularidade para escolha do limiar."""
    mat = np.array(embeddings, dtype=float)
    sim = cosine_similarity(mat)
    n = len(embeddings)
    req_ids = [f"R{i}" for i in range(n)]

    print()
    print(_c(BOLD, f"{'tau':>6}  {'arestas':>8}  {'comunidades':>12}  {'modularidade':>14}"))
    print(_c(DIM, "─" * 48))

    for tau in [0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.87, 0.90, 0.92, 0.95]:
        G = nx.Graph()
        G.add_nodes_from(req_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= tau:
                    G.add_edge(req_ids[i], req_ids[j], weight=float(sim[i, j]))

        if G.number_of_edges() == 0:
            print(f"{tau:>6.2f}  {'0':>8}  {'N/A':>12}  {'N/A':>14}")
            continue

        comms = louvain_communities(G, seed=42)
        q = nx.community.modularity(G, comms)
        print(f"{tau:>6.2f}  {G.number_of_edges():>8}  {len(comms):>12}  {q:>14.4f}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Constrói o Gabarito (Gold Standard) do TCC a partir do dataset de embeddings.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--topic", type=str, default="tarefas",
                        help="Tema para filtrar (ex: 'tarefas', 'restaurante'). Default: tarefas")
    parser.add_argument("--limit", type=int, default=50,
                        help="Máximo de user stories a carregar. Default: 50")
    parser.add_argument("--tau", type=float, default=0.82,
                        help="Limiar de similaridade de cosseno para criar arestas (0-1). Default: 0.82")
    parser.add_argument("--scan-tau", action="store_true",
                        help="Apenas escaneia limiares tau e imprime tabela (não persiste)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove o gabarito existente antes de recriar")
    args = parser.parse_args()

    print()
    print(_c(BOLD + CYAN, "╔══════════════════════════════════════════════════════╗"))
    print(_c(BOLD + CYAN, "║        TCC — Construção do Gabarito (Gold Standard)  ║"))
    print(_c(BOLD + CYAN, "╚══════════════════════════════════════════════════════╝"))
    print(f"  {_c(BOLD, 'Tema:')}    {args.topic}")
    print(f"  {_c(BOLD, 'Limite:')} {args.limit} user stories")
    print(f"  {_c(BOLD, 'Tau:')}    {args.tau}")
    print(f"  {_c(BOLD, 'CSV:')}    {config.CSV_PATH}")
    print()

    # 1. Carrega CSV
    df = _load_csv(config.CSV_PATH)

    # 2. Filtra tema
    df_topic = _filter_by_topic(df, args.topic, args.limit)

    # Parse embeddings
    print(_c(DIM, "  Parseando embeddings..."))
    embeddings_raw = [_parse_embedding(str(r)) for r in df_topic["embedding"]]
    # Remove linhas com embedding vazio
    valid_mask = [len(e) > 0 for e in embeddings_raw]
    df_topic = df_topic[valid_mask].reset_index(drop=True)
    embeddings = [e for e, v in zip(embeddings_raw, valid_mask) if v]
    print(_c(GREEN, f"  ✓ {len(embeddings)} user stories com embeddings válidos"))

    if len(embeddings) < 5:
        print(_c(RED, "  ✗ Embeddings insuficientes. Verifique o CSV."))
        return

    # 3. Scan de tau (sem persistência)
    if args.scan_tau:
        print(_c(BOLD, "\n  Escaneando limiares tau para o subconjunto filtrado:"))
        _scan_tau(embeddings)
        print(_c(DIM, "  Use --tau <valor> para construir o gabarito com o limiar escolhido."))
        return

    # 4. Conecta ao Neo4j
    print(_c(DIM, "Conectando ao Neo4j..."))
    client = Neo4jClient()
    client.test_connection()
    print(_c(GREEN, "✓ Neo4j conectado"))

    gid = _gabarito_graph_id(args.topic)
    print(f"  {_c(BOLD, 'graph_id:')} {gid}")
    print()

    # 5. Cleanup (opcional)
    if args.cleanup:
        _delete_gabarito(client, gid)
        print()

    # 6. Persiste nós
    t0 = time.time()
    print(_c(CYAN, "── [1/3] Persistindo nós no Neo4j..."))
    req_ids, embs = _save_nodes(client, gid, args.topic, df_topic)
    t1 = time.time()
    print(_c(DIM, f"  Tempo: {t1 - t0:.2f}s"))
    print()

    # 7. Cria arestas SIMILAR_TO
    print(_c(CYAN, "── [2/3] Calculando similaridade de cosseno e criando arestas..."))
    edge_count = _build_similarity_edges(client, gid, req_ids, embs, args.tau)
    t2 = time.time()
    print(_c(DIM, f"  Tempo: {t2 - t1:.2f}s"))
    print()

    # 8. Louvain
    print(_c(CYAN, "── [3/4] Rodando Louvain e gravando communityId..."))
    num_comms, modularity = _run_louvain(client, gid, req_ids, embs, args.tau)
    t3 = time.time()
    print(_c(DIM, f"  Tempo: {t3 - t2:.2f}s"))
    print()

    # 9. Conecta conceitos estáticos (Técnicas, Instruções)
    print(_c(CYAN, "── [4/4] Conectando nós estáticos (Técnicas, Instruções, Conceitos)..."))
    t4 = time.time()
    try:
        from src.infra.dspy.agent import _connect_graph_nodes
        _connect_graph_nodes(client, gid)
        print(_c(GREEN, "  ✓ Nós estáticos e relacionamentos padrão criados"))
    except Exception as e:
        print(_c(YELLOW, f"  ⚠ Erro ao conectar nós estáticos: {e}"))
    t5 = time.time()
    print(_c(DIM, f"  Tempo: {t5 - t4:.2f}s"))
    print()

    # ─── Resumo ───────────────────────────────────────────────────────────────
    total_time = round(t5 - t0, 2)
    print(_c(BOLD + GREEN, "═══ Gabarito Construído — Resumo ═══"))
    print(f"  graph_id:      {gid}")
    print(f"  Tema:          {args.topic}")
    print(f"  Nós:           {len(req_ids)}")
    print(f"  Arestas (τ≥{args.tau}): {edge_count}")
    print(f"  Comunidades:   {num_comms}")
    print(f"  Modularidade:  {modularity:.4f}")
    print(f"  Tempo total:   {total_time}s")
    print()
    print(_c(DIM, f"  Para avaliar os cenários contra este gabarito:"))
    print(_c(DIM, f"  python -m src.scripts.evaluate_graphs --topic \"{args.topic}\" --gabarito-gid {gid!r}"))
    print()

    # ─── Export JSON ─────────────────────────────────────────────────────────
    print(_c(CYAN, "── [5/5] Exportando JSON de estatísticas do Gabarito..."))
    
    rows_rel = client.run(
        "MATCH (:Requirement {graph_id:$gid})-[r]->() RETURN type(r) AS t, count(r) AS c",
        {"gid": gid}
    )
    rel_breakdown = {row["t"]: row["c"] for row in rows_rel}
    total_rels = sum(rel_breakdown.values())
    
    nodes = len(req_ids)
    graph_density = round(total_rels / (nodes * (nodes - 1)), 4) if nodes >= 2 else 0.0
    
    rows_dom = client.run("MATCH (r:Requirement {graph_id:$gid}) RETURN count(DISTINCT r.domain) AS c", {"gid": gid})
    unique_domains = rows_dom[0]["c"] if rows_dom else 1

    results = [{
        "scenario": "Gabarito",
        "run": 1,
        "graph_id": gid,
        "time_gen_s": 0.0,
        "time_rel_s": round(t5 - t0, 2),
        "time_total_s": round(t5 - t0, 2),
        "nodes": nodes,
        "llm_relationships": 0,
        "llm_rel_types": [],
        "total_relationships": total_rels,
        "rel_breakdown": rel_breakdown,
        "graph_density": graph_density,
        "unique_domains": unique_domains,
        "theme_adherence_pct": 100.0,
        "sample_req": "",
        "reqs_generated": nodes,
        "chain_of_thought_gen": [],
        "chain_of_thought_rel": [],
        "llm_rel_sample": []
    }]
    
    output = {
        "meta": {
            "topic": args.topic, "count": nodes, "runs": 1, "scenarios": ["Gabarito"],
            "model": "cosine_similarity_louvain",
            "timestamp": datetime.now().isoformat(),
            "note": "Gabarito gerado via CSV e Cosine Similarity"
        },
        "results": results
    }
    
    auto_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(auto_dir, exist_ok=True)
    save_path = os.path.join(auto_dir, "gabarito_stats.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(_c(GREEN, f"  ✓ Estatísticas salvas em {save_path}"))
    print()

    client.close()
    print(_c(BOLD + GREEN, "✓ Gabarito pronto!\n"))


if __name__ == "__main__":
    main()
