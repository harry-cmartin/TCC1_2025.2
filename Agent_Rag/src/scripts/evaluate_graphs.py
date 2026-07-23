#!/usr/bin/env python3
"""
Avaliação comparativa de criação de grafos — TCC.

Questão central: "Engenharia de prompt faz diferença na geração de grafos RAG?"

Gabarito: Gold Standard construído a partir do CSV (cosine + Louvain).
          Construído separadamente via: python -m src.scripts.build_gabarito

Cenário A: Modelo CRU — zero-shot, sem JSON, sem tipos, sem exemplos.
           O modelo decide o formato, a estrutura e as conexões.
Cenário B: Prompt BÁSICO — JSON estruturado mas apenas "RELATED_TO" como
           tipo de relacionamento. Sem few-shot, sem DSPy.
Cenário C: Prompt OTIMIZADO — especialista RE, JSON estruturado, exemplos reais,
           tipos de relacionamento (DEPENDS_ON, EXTENDS...), critérios e
           justificativas. Otimizado com DSPy.

Uso (a partir de Agent_Rag/):
    python3 -m src.scripts.evaluate_graphs --topic "tarefas" --count 20 --runs 1
    python3 -m src.scripts.evaluate_graphs --scenario A  # só cenário A (cru)
    python3 -m src.scripts.evaluate_graphs --scenario ABC --no-cleanup
    python3 -m src.scripts.evaluate_graphs --save-json results/tarefas.json
"""

import sys
import os
import argparse
import json
import time
import re
import textwrap
import statistics
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import dspy
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
from sklearn.metrics.pairwise import cosine_similarity

from src.infra.neo4j.client import Neo4jClient
from src import config

# ─── Cores ───────────────────────────────────────────────────────────────────
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


# ─── Configuração DSPy ────────────────────────────────────────────────────────

_LM_INSTANCE: Optional[dspy.LM] = None

def _setup_dspy() -> dspy.LM:
    """Cache DESABILITADO: garante respostas únicas por run (sem repetições)."""
    global _LM_INSTANCE
    _LM_INSTANCE = dspy.LM(
        model=config.DSPY_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        cache=False,
    )
    dspy.configure(lm=_LM_INSTANCE)
    return _LM_INSTANCE


# ─── Extração de Chain of Thought ────────────────────────────────────────────

def _extract_cot(result) -> str:
    r = getattr(result, "reasoning", None)
    return r.strip() if r and isinstance(r, str) else ""

def _extract_history_reasoning(lm: dspy.LM, before_len: int) -> str:
    history = getattr(lm, "history", [])
    parts: List[str] = []
    for entry in history[before_len:]:
        try:
            for choice in entry.get("response", {}).get("choices", []):
                msg = choice.get("message", {})
                rc = msg.get("reasoning_content", "")
                if rc and rc.strip():
                    parts.append(rc.strip())
                content = msg.get("content", "") or ""
                m = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
                if m:
                    parts.append(m.group(1).strip())
        except Exception:
            pass
    return "\n\n".join(parts)


# ─── Keywords de domínio para aderência ─────────────────────────────────────

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    # Dominio do dataset real (produtividade / gerenciamento de tarefas)
    "tarefas":       ["tarefa", "tarefas", "lista", "concluir", "prioridade", "prazo",
                      "lembrete", "notificacao", "notificação", "calendario", "calendário",
                      "evento", "agenda", "anotacao", "anotação", "organizar", "produtividade",
                      "projeto", "categoria", "filtro", "pesquisar", "pesquisa", "sincronizar"],
    "gerenciamento": ["gerenciamento", "gerenciar", "gerenciar", "sistema", "controle",
                      "gestao", "gestão", "administrar", "monitorar", "relatorio", "relatório",
                      "configurar", "cadastrar", "usuario", "usuário", "perfil", "acesso"],
    "produtividade": ["produtividade", "tarefa", "tarefas", "projeto", "prazo", "prioridade",
                      "calendario", "calendário", "lembrete", "notificacao", "notificação",
                      "anotacao", "anotação", "organizar", "sincronizar", "evento"],
    # Outros domínios suportados
    "restaurante":   ["restaurante", "cardapio", "cardápio", "pedido", "mesa", "garcom",
                      "garçom", "reserva", "prato", "cozinha", "cliente", "comida",
                      "delivery", "refeicao", "refeição", "menu", "estoque", "pagamento"],
    "banco":         ["banco", "conta", "transferencia", "transferência", "saldo",
                      "pagamento", "saque", "deposito", "depósito", "cartao", "cartão",
                      "pix", "extrato", "agencia", "agência", "cliente", "financeiro"],
    "hospital":      ["hospital", "paciente", "medico", "médico", "consulta", "exame",
                      "prontuario", "prontuário", "leito", "enfermagem", "prescricao",
                      "prescrição", "internacao", "internação", "laudo", "triagem"],
    "biblioteca":    ["biblioteca", "livro", "emprestimo", "empréstimo", "acervo",
                      "usuario", "devolucao", "devolução", "catalogo", "catálogo",
                      "reserva", "renovacao", "renovação"],
}

def _domain_kws(topic: str) -> List[str]:
    t = topic.lower().strip()
    for key, kws in _DOMAIN_KEYWORDS.items():
        if key in t or t in key:
            return kws
    return [kw for kw in t.split() if len(kw) > 2] or [t]


# ─── Parse JSON do LLM ───────────────────────────────────────────────────────

def _parse_json(raw: str) -> Any:
    """
    Parse seguro e tolerante para output cru do LLM.
    O Cenário B (prompt cru) pode retornar texto livre, não JSON.
    Tentamos várias estratégias antes de desistir.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    # Remove markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Tenta achar um array JSON
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Tenta achar um objeto JSON isolado
    m2 = re.search(r"\{.*\}", raw, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            return [obj] if isinstance(obj, dict) else obj
        except Exception:
            pass

    # Fallback: tenta parse direto
    try:
        return json.loads(raw)
    except Exception:
        raise ValueError(f"Não foi possível parsear JSON do output: {raw[:200]!r}")


# ─── Geração de requisitos ────────────────────────────────────────────────────

def _parse_raw_text_to_reqs(raw_text: str, limit: int) -> List[Dict]:
    """
    Converte texto livre (saída do cenário A cru) em lista de dicts de requisitos.
    Tenta extrair linhas numeradas ou com marcadores primeiro.
    """
    lines = []
    for ln in raw_text.split("\n"):
        ln = ln.strip()
        # Remove marcadores de lista e numeração
        ln = re.sub(r"^(\d+\.?|[-•*]|\*\*)\s*", "", ln)
        if len(ln) > 12:
            lines.append(ln[:250])
    if not lines:
        # fallback: pega o texto inteiro como um req
        lines = [raw_text[:250]]
    return [
        {"text": ln, "summary": "", "type": "funcional", "domain": "geral"}
        for ln in lines[:limit]
    ]


def _parse_raw_rels_to_list(
    raw_text: str,
    req_ids: List[str],
) -> List[Dict]:
    """
    Tenta extrair pares de IDs de um texto livre gerado no cenário A.
    Busca padrões como 'REQ_X ... REQ_Y' ou 'ID_A → ID_B'.
    """
    pairs: List[Dict] = []
    id_pattern = re.compile(r"(REQ_[A-Z0-9_]+)", re.IGNORECASE)
    ids_found = id_pattern.findall(raw_text)
    seen: set = set()
    for i in range(0, len(ids_found) - 1, 2):
        a, b = ids_found[i].upper(), ids_found[i + 1].upper()
        if a != b and (a, b) not in seen:
            pairs.append({"from": a, "to": b, "type": "RELATED_TO", "reason": "inferido (modelo cru)"})
            seen.add((a, b))
    return pairs


def _generate_reqs_a_raw(
    lm: dspy.LM,
    topic: str,
    count: int,
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Cenário A — MODELO CRUDÍSSIMO: zero-shot, sem JSON, sem tipos, sem exemplos.
    O LLM decide o formato, a língua, a estrutura — sem guia algum.
    Retorna (reqs, samples, reasonings).
    """
    from src.infra.dspy.signatures import GenerateGraphChunkRaw
    gen = dspy.ChainOfThought(GenerateGraphChunkRaw)
    h_before = len(getattr(lm, "history", []))
    try:
        res = gen(topic=f"{topic} (gere {count} requisitos)")
        cot = _extract_cot(res) or _extract_history_reasoning(lm, h_before)
        raw_output = getattr(res, "requirements_text", "") or ""
        reqs = _parse_raw_text_to_reqs(raw_output, count)
    except Exception as e:
        print(f"  {_c(YELLOW, f'[A-gen] falhou: {e}')}")
        reqs = [{"text": f"Requisito {i+1} sobre {topic}.", "summary": "",
                 "type": "funcional", "domain": "geral"} for i in range(count)]
        cot = ""
    samples = [reqs[0].get("text", "")[:150]] if reqs else []
    return reqs, samples, ([cot] if cot else [])


def _infer_relationships_a_raw(
    lm: dspy.LM,
    reqs_with_ids: List[Tuple[str, Dict]],
) -> Tuple[List[Dict], str]:
    """
    Cenário A — inferência de relacionamentos CRUA.
    Sem tipos, sem JSON — texto livre. O resultado será parseado de forma best-effort.
    """
    from src.infra.dspy.signatures import InferRelationshipsRaw
    items_str = "\n".join(f"[{rid}] {r.get('text', '')[:100]}" for rid, r in reqs_with_ids)
    infer = dspy.ChainOfThought(InferRelationshipsRaw)
    h_before = len(getattr(lm, "history", []))
    try:
        res = infer(items_list=items_str)
        cot = _extract_cot(res) or _extract_history_reasoning(lm, h_before)
        raw_text = getattr(res, "relations_text", "") or ""
        req_ids = [rid for rid, _ in reqs_with_ids]
        rels = _parse_raw_rels_to_list(raw_text, req_ids)
    except Exception as e:
        print(f"  {_c(YELLOW, f'[A-rel] inferência falhou: {e}')}")
        rels, cot = [], ""
    return rels, cot


def _generate_reqs_b(
    lm: dspy.LM,
    topic: str,
    count: int,
    chunk_size: int = 5,
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Cenário B — prompt BÁSICO com JSON mas apenas RELATED_TO.
    Retorna (reqs, samples, reasonings).
    """
    from src.infra.dspy.signatures import GenerateGraphChunkBasic
    gen = dspy.ChainOfThought(GenerateGraphChunkBasic)
    all_reqs, samples, reasonings = [], [], []
    remaining, chunk_num = count, 0

    while remaining > 0:
        batch = min(chunk_size, remaining)
        chunk_num += 1
        h_before = len(getattr(lm, "history", []))
        try:
            res = gen(project_name=topic, description=topic, count=str(batch))
            cot = _extract_cot(res) or _extract_history_reasoning(lm, h_before)
            if cot:
                reasonings.append(cot)
            raw_output = getattr(res, "requirements_json", "") or ""
            try:
                reqs = _parse_json(raw_output)
                if not isinstance(reqs, list):
                    raise ValueError("não é lista")
            except Exception:
                lines = [ln.strip() for ln in raw_output.split("\n") if ln.strip() and len(ln.strip()) > 10]
                reqs = [{"text": ln[:200], "summary": "", "type": "funcional", "domain": "geral"}
                        for ln in lines[:batch]]
                if not reqs:
                    reqs = [{"text": raw_output[:200], "summary": "", "type": "funcional", "domain": "geral"}]
            all_reqs.extend(reqs[:batch])
            if chunk_num == 1 and reqs:
                samples.append(reqs[0].get("text", "")[:150])
        except Exception as e:
            print(f"  {_c(YELLOW, f'[B-gen] chunk {chunk_num} falhou: {e}')}")
            for i in range(batch):
                all_reqs.append({"text": f"Req {len(all_reqs)+1} de {topic}.",
                                  "summary": "", "type": "funcional", "domain": "geral"})
        remaining -= batch

    return all_reqs, samples, reasonings


def _generate_reqs_c(
    lm: dspy.LM,
    client: Neo4jClient,
    topic: str,
    count: int,
    chunk_size: int = 5,
) -> Tuple[List[Dict], List[str], List[str]]:
    """Cenário C — prompt otimizado com exemplos. Retorna (reqs, samples, reasonings)."""
    from src.infra.dspy.signatures import GenerateGraphChunk
    kws = _domain_kws(topic)
    examples = client.sample_requirements_for_graph([k for k in kws if len(k) > 2][:5], min(10, count))
    if not examples:
        examples = client.sample_requirements_for_graph([], 10)
    reference_str = "\n".join(
        f"- {r['text']}" + (f" [Critérios: {r.get('summary','')[:60]}]" if r.get("summary") else "")
        for r in examples[:8]
    )

    gen = dspy.ChainOfThought(GenerateGraphChunk)
    all_reqs, samples, reasonings = [], [], []
    remaining, chunk_num = count, 0

    while remaining > 0:
        batch = min(chunk_size, remaining)
        chunk_num += 1
        h_before = len(getattr(lm, "history", []))
        try:
            res = gen(project_name=topic, description=topic, reference_examples=reference_str, count=str(batch))
            cot = _extract_cot(res) or _extract_history_reasoning(lm, h_before)
            if cot:
                reasonings.append(cot)
            reqs = _parse_json(res.requirements_json)
            all_reqs.extend(reqs[:batch])
            if chunk_num == 1 and reqs:
                samples.append(reqs[0].get("text", "")[:150])
        except Exception as e:
            print(f"  {_c(YELLOW, f'[C-gen] chunk {chunk_num} falhou: {e}')}")
            for i in range(batch):
                all_reqs.append({"text": f"Req {len(all_reqs)+1} de {topic}.", "summary": "", "type": "funcional", "domain": "geral"})
        remaining -= batch

    return all_reqs, samples, reasonings


# (Cenário A cru foi movido para _generate_reqs_a_raw acima)


# ─── Inferência de relacionamentos via LLM ────────────────────────────────────

def _reqs_to_list_str(reqs_with_ids: List[Tuple[str, Dict]]) -> str:
    """Formata a lista de requisitos para o prompt de inferência."""
    lines = []
    for req_id, r in reqs_with_ids:
        text = r.get("text", "")[:120]
        domain = r.get("domain", "geral")
        lines.append(f'  [{req_id}] ({domain}) {text}')
    return "\n".join(lines)


def _infer_relationships_b(
    lm: dspy.LM,
    reqs_with_ids: List[Tuple[str, Dict]],
) -> Tuple[List[Dict], str]:
    """
    Cenário B — LLM recebe prompt BÁSICO para inferir relacionamentos.
    Sem tipos, sem critérios, sem exemplos → resultado genérico.
    Retorna (lista_de_rels, reasoning).
    """
    from src.infra.dspy.signatures import InferRelationshipsBasic
    req_list_str = _reqs_to_list_str(reqs_with_ids)
    infer = dspy.ChainOfThought(InferRelationshipsBasic)
    h_before = len(getattr(lm, "history", []))
    try:
        res = infer(requirements_list=req_list_str)
        cot = _extract_cot(res) or _extract_history_reasoning(lm, h_before)
        rels = _parse_json(res.relationships_json)
        if not isinstance(rels, list):
            rels = []
    except Exception as e:
        print(f"  {_c(YELLOW, f'[B-rel] inferência falhou: {e}')}")
        rels, cot = [], ""
    return rels, cot


def _infer_relationships_c(
    lm: dspy.LM,
    reqs_with_ids: List[Tuple[str, Dict]],
    topic: str,
) -> Tuple[List[Dict], str]:
    """
    Cenário C — LLM recebe prompt OTIMIZADO com tipos, critérios e justificativa.
    Com guia de tipos (DEPENDS_ON, EXTENDS, CONFLICTS_WITH...) → resultado semântico.
    Retorna (lista_de_rels, reasoning).
    """
    from src.infra.dspy.signatures import InferRelationshipsOptimized
    req_list_str = _reqs_to_list_str(reqs_with_ids)
    infer = dspy.ChainOfThought(InferRelationshipsOptimized)
    h_before = len(getattr(lm, "history", []))
    try:
        res = infer(requirements_list=req_list_str, domain=topic)
        cot = _extract_cot(res) or _extract_history_reasoning(lm, h_before)
        rels = _parse_json(res.relationships_json)
        if not isinstance(rels, list):
            rels = []
    except Exception as e:
        print(f"  {_c(YELLOW, f'[C-rel] inferência falhou: {e}')}")
        rels, cot = [], ""
    return rels, cot


# ─── Persistência no Neo4j ────────────────────────────────────────────────────

def _save_requirements(
    client: Neo4jClient,
    gid: str,
    name: str,
    reqs: List[Dict],
) -> Tuple[List[Tuple[str, Dict]], List[List[float]]]:
    """Salva requisitos com embeddings locais e retorna lista de (req_id, req_dict)."""
    client.create_graph_meta(gid, name)
    safe = re.sub(r"[^a-z0-9]", "_", gid[:8])

    # Garante que o índice vetorial local existe
    try:
        from src.infra.embeddings import ensure_local_vector_index, embed_batch, LOCAL_EMBEDDING_MODEL
        ensure_local_vector_index(client)
        # Gera todos os embeddings em batch (muito mais rápido que um a um)
        texts = [req.get("text", "") for req in reqs]
        print(f"  [embeddings] Gerando {len(texts)} embeddings locais...")
        embeddings = embed_batch(texts)
        print(f"  [embeddings] ✓ {len(embeddings)} embeddings prontos")
        use_embeddings = True
    except Exception as e:
        print(f"  [embeddings] ⚠ Falha ao gerar embeddings: {e}. Salvando sem vetor.")
        embeddings = [[] for _ in reqs]
        use_embeddings = False

    reqs_with_ids: List[Tuple[str, Dict]] = []
    for i, req in enumerate(reqs):
        req_id = f"REQ_{safe}{i:03d}_{int(time.time()*100)%10**6:06d}"
        emb = embeddings[i] if use_embeddings else []
        client.create_requirement(
            req_id=req_id,
            text=req.get("text", ""),
            summary=req.get("summary", ""),
            req_type=req.get("type", "funcional"),
            domain=req.get("domain", "geral"),
            source="eval_generated",
            embedding=[],       # campo original (1536 dims OpenAI) — mantido vazio
            graph_id=gid,
        )
        # Salva o embedding local (384 dims) separadamente
        if use_embeddings and emb:
            client.set_local_embedding(req_id, emb)
        reqs_with_ids.append((req_id, req))
    return reqs_with_ids, embeddings


def _save_llm_relationships(
    client: Neo4jClient,
    gid: str,
    rels: List[Dict],
    req_id_map: Dict[str, str],  # idx_key -> real req_id
) -> int:
    """Salva os relacionamentos inferidos pelo LLM no Neo4j. Retorna count de rels criadas."""
    valid_types = {"DEPENDS_ON", "EXTENDS", "CONFLICTS_WITH", "RELATED_TO", "IMPLEMENTS"}
    count = 0
    for rel in rels:
        from_key = str(rel.get("from", "")).strip()
        to_key   = str(rel.get("to", "")).strip()
        rel_type = str(rel.get("type", "RELATED_TO")).strip().upper()
        reason   = str(rel.get("reason", "")).strip()[:200]

        if rel_type not in valid_types:
            rel_type = "RELATED_TO"

        # Mapeia o ID retornado pelo LLM → req_id real do Neo4j
        from_id = req_id_map.get(from_key)
        if not from_id:
            matches = [k for k in req_id_map.keys() if from_key and (k.startswith(from_key) or from_key in k)]
            if len(matches) == 1:
                from_id = req_id_map[matches[0]]

        to_id   = req_id_map.get(to_key)
        if not to_id:
            matches = [k for k in req_id_map.keys() if to_key and (k.startswith(to_key) or to_key in k)]
            if len(matches) == 1:
                to_id = req_id_map[matches[0]]

        if not from_id or not to_id or from_id == to_id:
            continue

        try:
            client.run(
                f"MATCH (a:Requirement {{req_id:$a, graph_id:$gid}}), "
                f"(b:Requirement {{req_id:$b, graph_id:$gid}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"ON CREATE SET r.reason=$reason, r.source='llm_inferred'",
                {"a": from_id, "b": to_id, "gid": gid, "reason": reason},
            )
            count += 1
        except Exception:
            pass
    return count

def _build_similarity_edges(
    client: Neo4jClient,
    gid: str,
    req_ids: List[str],
    embeddings: List[List[float]],
    tau: float,
) -> int:
    mat = np.array(embeddings, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2:
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
    return count

def _run_louvain(
    client: Neo4jClient,
    gid: str,
    req_ids: List[str],
    embeddings: List[List[float]],
    tau: float,
) -> Tuple[int, float]:
    mat = np.array(embeddings, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2:
        return 0, 0.0
    sim = cosine_similarity(mat)
    n = len(req_ids)
    G = nx.Graph()
    G.add_nodes_from(req_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= tau:
                G.add_edge(req_ids[i], req_ids[j], weight=float(sim[i, j]))
    if G.number_of_edges() == 0:
        return 0, 0.0
    communities = louvain_communities(G, seed=42)
    modularity = nx.community.modularity(G, communities)
    for comm_id, community in enumerate(communities):
        for req_id in community:
            client.run(
                "MATCH (r:Requirement {req_id:$req_id, graph_id:$gid}) SET r.communityId = $cid",
                {"req_id": req_id, "gid": gid, "cid": comm_id},
            )
    return len(communities), modularity


# ─── Métricas ─────────────────────────────────────────────────────────────────

def _count_nodes(client: Neo4jClient, gid: str) -> int:
    rows = client.run("MATCH (r:Requirement {graph_id:$gid}) RETURN count(r) AS c", {"gid": gid})
    return rows[0]["c"] if rows else 0

def _count_rels_by_type(client: Neo4jClient, gid: str) -> Dict[str, int]:
    rows = client.run(
        "MATCH (:Requirement {graph_id:$gid})-[r]->() RETURN type(r) AS t, count(r) AS c",
        {"gid": gid},
    )
    return {row["t"]: row["c"] for row in rows}

def _graph_density(client: Neo4jClient, gid: str) -> float:
    """Densidade do grafo = rels / (n*(n-1)) onde n = número de nós."""
    n = _count_nodes(client, gid)
    if n < 2:
        return 0.0
    rows = client.run("MATCH (:Requirement {graph_id:$gid})-[r]->() RETURN count(r) AS c", {"gid": gid})
    total_rels = rows[0]["c"] if rows else 0
    return round(total_rels / (n * (n - 1)), 4)

def _theme_adherence(client: Neo4jClient, gid: str, topic: str) -> float:
    keywords = _domain_kws(topic)
    rows = client.run("MATCH (r:Requirement {graph_id:$gid}) RETURN toLower(r.text) AS txt", {"gid": gid})
    if not rows:
        return 0.0
    hits = sum(1 for r in rows if any(kw in r["txt"] for kw in keywords))
    return round(hits / len(rows) * 100, 1)

def _count_unique_domains(client: Neo4jClient, gid: str) -> int:
    rows = client.run("MATCH (r:Requirement {graph_id:$gid}) RETURN count(DISTINCT r.domain) AS c", {"gid": gid})
    return rows[0]["c"] if rows else 0

def _delete_graph(client: Neo4jClient, gid: str) -> None:
    client.run("MATCH (r:Requirement {graph_id:$gid}) DETACH DELETE r", {"gid": gid})
    client.run("MATCH (g:Graph {graph_id:$gid}) DETACH DELETE g", {"gid": gid})


# ─── Execução de um cenário ───────────────────────────────────────────────────

def _run_scenario(
    scenario: str,
    lm: dspy.LM,
    client: Neo4jClient,
    topic: str,
    count: int,
    run_idx: int,
    cleanup: bool,
) -> Dict[str, Any]:
    ts = int(time.time())
    safe_topic = re.sub(r"[^a-z0-9]", "_", topic.lower())[:12]
    gid  = f"eval_{scenario.lower()}_{safe_topic}_{run_idx}_{ts}"
    name = f"[Eval-{scenario}] {topic.title()} (run {run_idx + 1})"
    print(f"  {_c(DIM, f'gid: {gid}')}")

    # ── 1. Gera requisitos ────────────────────────────────────────────────────
    t0 = time.time()
    if scenario == "A":
        reqs, samples, gen_cots = _generate_reqs_a_raw(lm, topic, count)
    elif scenario == "B":
        reqs, samples, gen_cots = _generate_reqs_b(lm, topic, count)
    else:
        reqs, samples, gen_cots = _generate_reqs_c(lm, client, topic, count)
    gen_time = round(time.time() - t0, 2)
    print(f"  {_c(DIM, f'  geração: {gen_time}s ({len(reqs)} req)')} ")

    # ── 2. Salva nós no Neo4j ─────────────────────────────────────────────────
    t1 = time.time()
    reqs_with_ids, embs = _save_requirements(client, gid, name, reqs)
    req_id_map: Dict[str, str] = {}
    for real_id, _ in reqs_with_ids:
        req_id_map[real_id] = real_id
    
    # ── Executa Similaridade e Louvain para geração de comunidades ────────────
    print(f"  {_c(CYAN, '  → detectando comunidades (Louvain)...')}")
    req_ids = [r_id for r_id, _ in reqs_with_ids]
    _build_similarity_edges(client, gid, req_ids, embs, tau=0.82)
    _run_louvain(client, gid, req_ids, embs, tau=0.82)

    save_req_time = round(time.time() - t1, 2)

    # ── 3. Conecta conceitos estáticos (regras fixas) ──────────────────────────
    from src.infra.dspy.agent import _connect_graph_nodes
    _connect_graph_nodes(client, gid)

    # ── 4. Infere relacionamentos via LLM ─────────────────────────────────────
    t2 = time.time()
    if scenario == "A":
        # Cenário A cru — tenta inferir rels com texto livre
        print(f"  {_c(CYAN, '  → inferindo relacionamentos (modelo cru)...')}")
        llm_rels, rel_cot = _infer_relationships_a_raw(lm, reqs_with_ids)
    elif scenario == "B":
        print(f"  {_c(CYAN, '  → inferindo relacionamentos com o LLM (RELATED_TO)...')}")
        llm_rels, rel_cot = _infer_relationships_b(lm, reqs_with_ids)
    else:
        print(f"  {_c(CYAN, '  → inferindo relacionamentos com o LLM (tipos otimizados)...')}")
        llm_rels, rel_cot = _infer_relationships_c(lm, reqs_with_ids, topic)
    rel_time = round(time.time() - t2, 2)

    # ── 5. Salva relacionamentos no Neo4j ─────────────────────────────────────
    saved_rels = _save_llm_relationships(client, gid, llm_rels, req_id_map)
    total_time = round(gen_time + save_req_time + rel_time, 2)

    # ── 6. Coleta métricas ────────────────────────────────────────────────────
    nodes      = _count_nodes(client, gid)
    rels_by_type = _count_rels_by_type(client, gid)
    total_rels = sum(rels_by_type.values())
    density    = _graph_density(client, gid)
    adherence  = _theme_adherence(client, gid, topic)
    domains    = _count_unique_domains(client, gid)

    # Tipos distintos de relacionamento inferidos pelo LLM
    llm_rel_types = list({r.get("type", "RELATED_TO") for r in llm_rels})

    result: Dict[str, Any] = {
        "scenario": scenario,
        "run": run_idx + 1,
        "graph_id": gid,
        "time_gen_s": gen_time,
        "time_rel_s": rel_time,
        "time_total_s": total_time,
        "nodes": nodes,
        "llm_relationships": saved_rels,          # rels criadas pelo LLM
        "llm_rel_types": llm_rel_types,            # tipos distintos usados
        "total_relationships": total_rels,         # total no grafo
        "rel_breakdown": rels_by_type,             # breakdown por tipo
        "graph_density": density,
        "unique_domains": domains,
        "theme_adherence_pct": adherence,
        "sample_req": samples[0] if samples else "",
        "reqs_generated": len(reqs),
        "chain_of_thought_gen": gen_cots,         # CoT da geração
        "chain_of_thought_rel": [rel_cot] if rel_cot else [],  # CoT da inferência
        "llm_rel_sample": llm_rels[:5],           # amostra de rels para o JSON
    }

    if cleanup:
        _delete_graph(client, gid)
        print(f"  {_c(DIM, '  grafo removido.')}")

    return result


# ─── Tabela de resultados ─────────────────────────────────────────────────────

def _stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std  = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(mean, 2), round(std, 2)


_SCENARIO_LABELS = {
    "A": "A — Modelo Cru (zero-shot, sem guia)",
    "B": "B — Prompt Básico (JSON + RELATED_TO)",
    "C": "C — DSPy Otimizado (tipos + exemplos + CoT)",
}

_HUMAN_EFFORT_TABLE = {
    "A": {
        "setup_horas": "~0",
        "prompt_engineering": "Nenhum — prompt mínimo de 1 linha",
        "iteracoes_prompt": 0,
        "codigo_extra": "Apenas chamar o LLM com topic -> text",
        "nivel": "Baixo",
    },
    "B": {
        "setup_horas": "~1-2h",
        "prompt_engineering": "Manual — instrução de JSON + definição de RELATED_TO",
        "iteracoes_prompt": 3,
        "codigo_extra": "Signature DSPy + parser JSON",
        "nivel": "Médio",
    },
    "C": {
        "setup_horas": "~8-16h",
        "prompt_engineering": "Otimização via DSPy (compilação de few-shot, tipos semânticos, critérios)",
        "iteracoes_prompt": 10,
        "codigo_extra": "DSPy ReAct + Signatures + compiled_agent.json + infer_relationships_optimized",
        "nivel": "Alto",
    },
}


def _print_effort_table() -> None:
    """Imprime a tabela de nível de esforço humano para cada cenário."""
    print()
    print(_c(BOLD + BLUE, "═══ Nível de Esforço Humano por Cenário ═══"))
    print()
    headers = ["Cenário", "Nível", "Setup (h)", "Iter. Prompt", "Prompt Engineering"]
    col_w   = [38, 9, 11, 14, 55]
    print(_c(BOLD, "  ".join(h.ljust(w) for h, w in zip(headers, col_w))))
    print(_c(DIM,  "─" * 135))
    colors = {"A": RED, "B": YELLOW, "C": GREEN}
    for sc in ["A", "B", "C"]:
        e = _HUMAN_EFFORT_TABLE[sc]
        row = [
            _SCENARIO_LABELS[sc],
            e["nivel"],
            e["setup_horas"],
            str(e["iteracoes_prompt"]),
            e["prompt_engineering"],
        ]
        print(_c(colors[sc], "  ".join(v.ljust(w) for v, w in zip(row, col_w))))
    print()
    print(_c(DIM, "  * Esforço humano é estimativo e pode ser ajustado manualmente no código."))


def _print_summary(results: List[Dict], topic: str, count: int, runs: int) -> None:
    by_sc: Dict[str, List[Dict]] = {"A": [], "B": [], "C": []}
    for r in results:
        if r["scenario"] in by_sc:
            by_sc[r["scenario"]].append(r)

    print()
    print(_c(BOLD + CYAN, f"═══ Avaliação: {topic!r}  ({count} nós, {runs} execuções) ═══"))
    print()

    headers = ["Cenário", "Tempo (s)", "Nós", "Tot-Rels", "LLM-Rels", "Tipos", "Domínios", "Aderência"]
    col_w   = [38, 13, 7, 10, 10, 7, 10, 12]
    print(_c(BOLD, "  ".join(h.ljust(w) for h, w in zip(headers, col_w))))
    print(_c(DIM,  "─" * 115))

    rows_data = []
    for sc in ["A", "B", "C"]:
        label = _SCENARIO_LABELS[sc]
        sc_r = by_sc[sc]
        if not sc_r:
            continue
        t_m, t_s    = _stats([r["time_total_s"] for r in sc_r])
        n_m, _      = _stats([float(r["nodes"]) for r in sc_r])
        trl_m, trl_s= _stats([float(r["total_relationships"]) for r in sc_r])
        rl_m, rl_s  = _stats([float(r["llm_relationships"]) for r in sc_r])
        tp_m, _     = _stats([float(len(r["llm_rel_types"])) for r in sc_r])
        dm_m, _     = _stats([float(r["unique_domains"]) for r in sc_r])
        ad_m, ad_s  = _stats([r["theme_adherence_pct"] for r in sc_r])

        color = RED if sc == "A" else (YELLOW if sc == "B" else GREEN)
        row   = [label, f"{t_m} ±{t_s}", f"{n_m:.0f}",
                 f"{trl_m:.1f} ±{trl_s:.1f}", f"{rl_m:.1f} ±{rl_s:.1f}", f"{tp_m:.0f}", f"{dm_m:.0f}", f"{ad_m:.0f}% ±{ad_s:.0f}%"]
        rows_data.append((sc, rl_m, tp_m, ad_m))
        print(_c(color, "  ".join(v.ljust(w) for v, w in zip(row, col_w))))

    print()
    if len(rows_data) >= 2:
        sc_c = next((r for r in rows_data if r[0] == "C"), None)
        baseline = next((r for r in rows_data if r[0] == "A"), None)

        if sc_c and baseline:
            c_rl, c_tp, c_ad = sc_c[1], sc_c[2], sc_c[3]
            b_rl, b_tp, b_ad = baseline[1], baseline[2], baseline[3]

            wins = []
            if c_rl > b_rl: wins.append(f"+{c_rl-b_rl:.1f} relacionamentos")
            if c_tp > b_tp: wins.append(f"+{c_tp-b_tp:.0f} tipos distintos")
            if c_ad > b_ad: wins.append(f"+{c_ad-b_ad:.0f}% aderência")
            if wins:
                print(_c(GREEN, f"  ✓ DSPy Otimizado (C): {', '.join(wins)} vs Modelo Cru (A)"))
            else:
                print(_c(YELLOW, "  ⚠ Diferença pequena neste run. Tente --runs 3 para mais estabilidade."))


def _wrap(text: str, width: int = 68, indent: str = "  │      ") -> str:
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for line in textwrap.wrap(para, width - len(indent)):
            lines.append(indent + line)
    return "\n".join(lines)


def _print_explainer(results: List[Dict]) -> None:
    """
    Exibe side-by-side:
    1. Chain of Thought da GERAÇÃO (como o modelo pensou pra criar os requisitos)
    2. Chain of Thought da INFERÊNCIA (como o modelo pensou pra criar os relacionamentos)
    3. Amostra dos relacionamentos criados (com types e reasons no Cenário C)
    """
    print()
    print(_c(BOLD + MAGENTA, "═══ Chain of Thought — Raciocínio Comparativo B vs C ═══"))
    print()

    by_sc: Dict[str, List[Dict]] = {"A": [], "B": [], "C": []}
    for r in results:
        if r["scenario"] in by_sc:
            by_sc[r["scenario"]].append(r)

    for sc, label, color in [("A", "Modelo Cru (zero-shot, sem instruções)", RED),
                               ("B", "Prompt Básico (JSON + apenas RELATED_TO)", YELLOW),
                               ("C", "DSPy Otimizado (exemplos + tipos + critérios + compilado)", GREEN)]:
        sc_results = by_sc[sc]
        if not sc_results:
            continue

        print(_c(color + BOLD, f"  ┌─ [{sc}] {label}"))

        for r in sc_results:
            rl_types = r.get("llm_rel_types", [])
            rl_sample = r.get("llm_rel_sample", [])

            print()
            print(_c(color, f"  │  Run {r['run']} ─ "
                            f"{r['time_gen_s']}s geração + {r['time_rel_s']}s inferência | "
                            f"{r['nodes']} nós | "
                            f"{r['llm_relationships']} LLM-rels | "
                            f"{r['theme_adherence_pct']}% aderência"))

            # Requisito de amostra
            sample = r.get("sample_req", "")
            if sample:
                print(_c(DIM, f"  │  Req gerado: \"{sample[:100]}{'...' if len(sample)>100 else ''}\""))

            # CoT da GERAÇÃO
            gen_cots: List[str] = r.get("chain_of_thought_gen", [])
            if gen_cots:
                print(_c(color, "  │"))
                print(_c(color, "  │  🧠 Raciocínio ao GERAR os requisitos:"))
                cot_display = gen_cots[0][:500] + ("..." if len(gen_cots[0]) > 500 else "")
                print(_c(DIM, _wrap(cot_display)))

            # CoT da INFERÊNCIA DE RELACIONAMENTOS
            rel_cots: List[str] = r.get("chain_of_thought_rel", [])
            if rel_cots:
                print(_c(color, "  │"))
                print(_c(color, "  │  🔗 Raciocínio ao INFERIR os relacionamentos:"))
                cot_display = rel_cots[0][:600] + ("..." if len(rel_cots[0]) > 600 else "")
                print(_c(DIM, _wrap(cot_display)))
            else:
                print(_c(DIM, "  │  🔗 Raciocínio de inferência: não capturado"))

            # Tipos de rels usados
            if rl_types:
                print(_c(color, f"  │  Tipos de relação usados: {', '.join(rl_types)}"))

            # Amostra dos relacionamentos
            if rl_sample:
                print(_c(color, f"  │  Amostra de relacionamentos LLM-inferidos:"))
                for rel in rl_sample[:4]:
                    rel_type = rel.get("type", "RELATED_TO")
                    reason   = rel.get("reason", "")
                    from_id  = rel.get("from", "?")[:20]
                    to_id    = rel.get("to", "?")[:20]
                    line = f"  │    [{from_id}] --{rel_type}--> [{to_id}]"
                    if reason:
                        line += f": {reason[:80]}"
                    print(_c(DIM, line))

        print(_c(color, f"  └{'─'*62}"))
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Avaliação B vs C: LLM gera requisitos E infere relacionamentos.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--topic",     type=str,
                        default="gerenciamento de tarefas e produtividade",
                        help="Tema (condizente com o dataset: tarefas, eventos, calendario, anotacoes)")
    parser.add_argument("--count",     type=int, default=20)
    parser.add_argument("--runs",      type=int, default=3)
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--scenario",  type=str, default="ABC", help="Cenários: A, B, C, ou combinação (ex: ABC)")
    args = parser.parse_args()

    count    = min(max(args.count, 5), 100)
    cleanup  = not args.no_cleanup
    scenarios = list(args.scenario)

    print()
    print(_c(BOLD + CYAN, "╔══════════════════════════════════════════════════════════════╗"))
    print(_c(BOLD + CYAN, "║       TCC — Avaliação de Prompt Engineering (Grafos RAG)     ║"))
    print(_c(BOLD + CYAN, "║  A = Cru (zero-shot) | B = RELATED_TO | C = DSPy Otimizado   ║"))
    print(_c(BOLD + CYAN, "╚══════════════════════════════════════════════════════════════╝"))
    print(f"  {_c(BOLD, 'Questão:')} Engenharia de prompt faz diferença na geração de grafos RAG?")
    print(f"  {_c(RED,    'A (cru):')}    Zero-shot — sem JSON, sem tipos, sem exemplos. O modelo decide tudo.")
    print(f"  {_c(YELLOW, 'B (básico):')} JSON + RELATED_TO — estrutura definida, tipo de rel. único.")
    print(f"  {_c(GREEN,  'C (DSPy):')}   Especialista RE + JSON + exemplos + tipos + critérios + compilado.")
    print()
    print(f"  {_c(BOLD, 'Tema:')}       {args.topic}")
    print(f"  {_c(BOLD, 'Nós/grafo:')} {count}")
    print(f"  {_c(BOLD, 'Runs:')}       {args.runs} por cenário")
    print(f"  {_c(BOLD, 'Cenários:')}  {' e '.join(scenarios)}")
    print(f"  {_c(BOLD, 'Cleanup:')}   {'Sim' if cleanup else 'Não'}")
    print(f"  {_c(BOLD, 'Cache LLM:')} Desabilitado")
    print()

    print(_c(DIM, "Conectando ao Neo4j..."))
    client = Neo4jClient()
    client.test_connection()
    print(_c(GREEN, "✓ Neo4j conectado"))

    print(_c(DIM, "Configurando DSPy/LLM..."))
    lm = _setup_dspy()
    print(_c(GREEN, f"✓ LLM: {config.DSPY_MODEL}"))
    print()

    all_results: List[Dict] = []
    total_runs  = len(scenarios) * args.runs
    run_counter = 0

    for sc in scenarios:
        sc = sc.upper()
        if sc not in ["A", "B", "C"]: continue
        
        sc_label = _SCENARIO_LABELS.get(sc, sc)
        print(_c(BOLD, f"\n── Cenário {sc_label}"))

        for i in range(args.runs):
            run_counter += 1
            pct = int(run_counter / total_runs * 100)
            print(f"\n  {_c(CYAN, f'Run {i+1}/{args.runs}')} [{pct:3d}%] "
                  f"gerando {count} req. sobre '{args.topic}'...")

            result = _run_scenario(
                scenario=sc, lm=lm, client=client,
                topic=args.topic, count=count, run_idx=i, cleanup=cleanup,
            )
            all_results.append(result)
            cot_ok  = "🧠" if result.get("chain_of_thought_gen") else "○"
            rel_cot = "🔗" if result.get("chain_of_thought_rel") else "○"
            types   = ", ".join(result.get("llm_rel_types", [])) or "nenhum"
            print(
                f"  {_c(GREEN, '✓')} {result['time_total_s']}s | "
                f"{result['nodes']} nós | "
                f"{result['llm_relationships']} LLM-rels [{types}] | "
                f"{result['theme_adherence_pct']}% aderência | {cot_ok}CoT-gen {rel_cot}CoT-rel"
            )
        print()

    _print_summary(all_results, args.topic, count, args.runs)
    _print_explainer(all_results)
    _print_effort_table()

    # ─── Salva JSON ───────────────────────────────────────────────────────────
    output = {
        "meta": {
            "topic": args.topic, "count": count, "runs": args.runs,
            "scenarios": scenarios, "model": config.DSPY_MODEL,
            "timestamp": datetime.now().isoformat(),
            "note": "A=Modelo Cru(zero-shot) | B=Básico(RELATED_TO) | C=DSPy Otimizado",
            "scenario_descriptions": _SCENARIO_LABELS,
            "human_effort": _HUMAN_EFFORT_TABLE,
        },
        "results": all_results,
    }

    if args.save_json:
        save_path = args.save_json
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    else:
        auto_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "results")
        os.makedirs(auto_dir, exist_ok=True)
        stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(auto_dir, f"avaliacao_{args.topic}_{stamp}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(_c(GREEN, f"\n✓ Resultados salvos em: {save_path}"))

    client.close()
    print(_c(BOLD + GREEN, "✓ Avaliação concluída!\n"))


if __name__ == "__main__":
    main()
