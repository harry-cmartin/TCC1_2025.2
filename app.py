import os
import sys

# Garante que a pasta Agent_Rag seja resolvida corretamente 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "Agent_Rag")))

from src.api.main import app

# O Hugging Face Gradio SDK irá detectar a variável "app" (FastAPI) e rodá-la usando Uvicorn na porta 7860 automaticamente.
