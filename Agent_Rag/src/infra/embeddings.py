"""
Módulo de embeddings locais via sentence-transformers.

Usa o modelo multilíngue (suporta Português) sem nenhuma API key.
O modelo é baixado automaticamente na primeira execução (~90MB).

Dimensões: 384 (compatível com o índice Neo4j LOCAL_EMBEDDING_DIMS)
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

# Constantes públicas — usadas para criar o índice vetorial no Neo4j
LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBEDDING_DIMS  = 384
LOCAL_INDEX_NAME      = "requirement_embeddings_local"

_model = None  # carregamento lazy


def _get_model():
    """Carrega o modelo sentence-transformers na primeira chamada (lazy)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers não instalado. Execute:\n"
                "  pip install sentence-transformers\n"
            )
        print(f"[embeddings] Carregando modelo '{LOCAL_EMBEDDING_MODEL}'... (1ª vez faz download ~90MB)")
        _model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print(f"[embeddings] ✓ Modelo pronto | dims={LOCAL_EMBEDDING_DIMS}")
    return _model


def embed_text(text: str) -> List[float]:
    """Gera embedding para um único texto. Retorna lista de floats (384 dims)."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist()


def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Gera embeddings para uma lista de textos. Mais eficiente que chamar embed_text em loop."""
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 20,
    )
    return [v.tolist() for v in vecs]


def ensure_local_vector_index(client) -> None:
    """Garante que o índice vetorial local (384 dims) existe no Neo4j."""
    try:
        client.run(
            f"""
            CREATE VECTOR INDEX {LOCAL_INDEX_NAME} IF NOT EXISTS
            FOR (r:Requirement) ON (r.embedding_local)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {LOCAL_EMBEDDING_DIMS},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
        print(f"[embeddings] ✓ Índice vetorial '{LOCAL_INDEX_NAME}' garantido no Neo4j")
    except Exception as e:
        print(f"[embeddings] ⚠ Não foi possível criar índice vetorial: {e}")
