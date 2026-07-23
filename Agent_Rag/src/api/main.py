#!/usr/bin/env python3
"""
Entry point da API — FastAPI + Uvicorn.

Executar a partir da raiz do projeto Agent_Rag/:
    python -m src.api.main
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Garante imports absolutos a partir da raiz do Agent_Rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src import config  # noqa: E402
from src.infra.neo4j.client import Neo4jClient  # noqa: E402
from src.infra.dspy.agent import build_agent  # noqa: E402
from src.service.graph_service import GraphService  # noqa: E402
from src.service.chat_service import ChatService  # noqa: E402
from src.api.routes import chat as chat_routes, graph as graph_routes, ws as ws_routes  # noqa: E402


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("=" * 60)
        print("  RE Expert Agent — Startup")
        print("=" * 60)

        # 1. Neo4j
        try:
            client = Neo4jClient()
            client.test_connection()
        except Exception as e:
            print(f"\n[ERRO] Nao foi possivel conectar ao Neo4j ({config.NEO4J_URL}).")
            print(f"       {e}")
            sys.exit(1)

        print(f"[ok] Neo4j conectado ({config.NEO4J_URL})")

        # 2. Serviços
        graph_service = GraphService(client)
        graph_routes.init(graph_service)

        # 3. Migração: tagging de nós sem graph_id + meta do grafo default
        client.set_missing_graph_ids("default")
        client.create_graph_meta("default", "Grafo Principal")

        # 4. Popular grafo se vazio
        n = client.node_count()
        if n == 0:
            if not os.path.exists(config.CSV_PATH):
                print(f"[ERRO] CSV nao encontrado: {config.CSV_PATH}")
                sys.exit(1)
            print("[!] Grafo vazio — populando automaticamente...")
            graph_service.populate_complete()
            n = client.node_count()

        print(f"[ok] Grafo: {n} nos")

        # 4b. Grafo de teste automático (se SEED_TEST_GRAPH_NAME estiver definido)
        if config.SEED_TEST_GRAPH_NAME:
            existing = client.list_graphs()
            already_exists = any(g["name"] == config.SEED_TEST_GRAPH_NAME for g in existing)
            if not already_exists:
                import time as _t
                test_gid = f"test_{int(_t.time())}"
                print(f"[!] Criando grafo de teste '{config.SEED_TEST_GRAPH_NAME}' ({config.SEED_TEST_GRAPH_COUNT} nos)...")
                graph_service.populate_graph_from_dataset(
                    test_gid, config.SEED_TEST_GRAPH_NAME, count=config.SEED_TEST_GRAPH_COUNT
                )
                print(f"[ok] Grafo de teste criado: graph_id={test_gid}")
            else:
                print(f"[ok] Grafo de teste '{config.SEED_TEST_GRAPH_NAME}' ja existe.")

        # 5. Agente DSPy
        agent = build_agent(client)
        chat_service = ChatService(agent)
        chat_routes.init(chat_service)
        ws_routes.init(chat_service)

        print("[ok] Servidor pronto em http://localhost:8000")
        print("[ok] Frontend esperado em http://localhost:5173")
        print("=" * 60)

        yield  # aplicação rodando

    app = FastAPI(title="RE Expert Agent", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(graph_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(ws_routes.router)

    # Configuração para servir o frontend (Vite SPA) no mesmo servidor (ex: Hugging Face Spaces)
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist"))
    if os.path.exists(static_dir):
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        # Serve a pasta /assets (CSS/JS do Vite)
        assets_dir = os.path.join(static_dir, "assets")
        if os.path.exists(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        # Rota catch-all para servir arquivos estáticos da raiz ou o index.html (SPA)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file_path = os.path.join(static_dir, full_path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[os.path.join(config.PROJECT_ROOT, "src")],
    )
