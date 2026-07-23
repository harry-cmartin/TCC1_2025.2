import os
import json
from logger_config import get_logger
from schemas.models import APIResponse
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from .OTel.tracing import setup_tracing
from .OTel.decorators import trace_async, span, trace_async_generator
from .Langfuse import langfuse_trace, langfuse_trace_generator, langfuse
from api.Langfuse.decorators import observe
from api.mirror.decorators import mirror_trace, start_trace_session
from .OTel.costs import estimate_cost
from .mirror.decorators import _mirror_tracer# importa o tracer
setup_tracing()

logger = get_logger(__name__)

# =====================================
# TOOL DEFINITIONS
# =====================================

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_qdrant",
            "description": "Buscar informações na base de conhecimento da empresa Servopa sobre veículos, consórcio, financiamento e serviços.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Pergunta do usuário"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_canvas",
            "description": (
                "Abre um canvas (painel lateral) e escreve conteúdo longo e formatado nele. "
                "Use SEMPRE que o usuário pedir para: criar, escrever, gerar, elaborar, documentar, "
                "redigir, produzir ou compor qualquer texto, artigo, resumo ou documento. "
                "Exemplos: 'escreva sobre cachorro', 'abre um canvas', 'crie um texto sobre X', "
                "'elabore sobre Y', 'me dê um artigo sobre Z'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Título curto para o canvas"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Instrução completa do que deve ser escrito no canvas"
                    }
                },
                "required": ["title", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_stats_widget",
            "description": (
                "Exibe um card de indicador inline no chat. "
                "Use SOMENTE para Qdrant ou Redis — nenhum outro banco de dados. "
                "Para 'chunks/registros ativos no total', chame esta ferramenta DUAS VEZES: "
                "uma com qdrant_enabled e outra com redis_enabled. "
                "Se o usuário citar um banco desconhecido, use 'answer' para informar quais estão disponíveis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "widget": {
                        "type": "string",
                        "enum": [
                            "qdrant_enabled",
                            "qdrant_disabled",
                            "qdrant_total",
                            "redis_enabled",
                            "redis_disabled",
                            "redis_total"
                        ],
                        "description": "Indicador que deve ser exibido"
                    }
                },
                "required": ["widget"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": (
                "Responde diretamente ao usuário. "
                "Use para: saudações, informações sobre capacidades, pedidos inválidos/ambíguos, "
                "ou perguntas fora do escopo. Sempre que a resposta for texto livre (não canvas), "
                "use esta ferramenta. NUNCA responda como texto livre sem usar a ferramenta."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Sua resposta"
                    }
                },
                "required": ["text"]
            }
        }
    },
]

_WIDGET_LABELS = {
    "qdrant_total": "Qdrant Total",
    "qdrant_enabled": "Qdrant Ativos",
    "qdrant_disabled": "Qdrant Inativos",
    "redis_total": "Redis Total",
    "redis_enabled": "Redis Ativos",
    "redis_disabled": "Redis Inativos",
}


class ChatClient:

    def __init__(self, client_qdrant=None, client_edgedb=None):

        self.client_qdrant = client_qdrant
        self.client_edgedb = client_edgedb

        self._azure_client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "VectorHub"),
            },
        )

        self._deployment = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self._embedding_deployment = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

        self._qdrant = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

    # =====================================
    # EMBEDDING
    # =====================================
    async def _llm_call(self, messages, **kwargs):
        import time

        model=kwargs.pop("model", self._deployment)
        if kwargs.get("stream"):
            return await self._azure_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        start = time.time()

        response = await self._azure_client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

        duration = time.time() - start

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        cost = estimate_cost(self._deployment, usage) if usage else None

        msg = response.choices[0].message

        content = msg.content

        tool_data = None

        if msg.tool_calls:
            try:
                tool_data = [
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in msg.tool_calls
                ]
            except:
                tool_data = str(msg.tool_calls)

        _mirror_tracer.log_custom_event("llm.call", {
            "model": self._deployment,
            "messages": messages,
            "response": content,
            "tool_calls": tool_data,
            "usage": usage,
            "cost": cost,
            "duration_ms": round(duration * 1000, 2)
        })

        return response

    # =====================================
    # EMBEDDING
    # =====================================

    @mirror_trace("embedding.create")
    @observe("embedding.create")
    @trace_async("embedding.create")
    async def _generate_embedding(self, text: str):

        response = await self._azure_client.embeddings.create(
            model=self._embedding_deployment,
            input=text
        )

        return response.data[0].embedding

    # =====================================
    # TOOL: QDRANT SEARCH
    # =====================================
    @mirror_trace("qdrant.search")
    @observe("qdrant.search")
    @trace_async("qdrant.search")
    async def _buscar_qdrant(self, question: str, collection: str):
        with span("qdrant.query", collection=collection):

            embedding = await self._generate_embedding(question)

            results = await self._qdrant.search(
                collection_name=collection,
                query_vector=("pageContent", embedding),
                limit=5,
                with_payload=True
            )

            texts = [
                r.payload.get("pageContent")
                for r in results
                if r.payload and r.payload.get("pageContent")
            ]

            return "\n\n".join(texts)

    # =====================================
    # SELF EVALUATION
    # =====================================
    @mirror_trace("llm.self_evaluation")
    @observe("llm.self_evaluation")
    @trace_async("llm.self_evaluation")
    async def _avaliar_resposta(self, question: str, answer: str):

        prompt = f"""
Pergunta do usuário:
{question}

Resposta gerada:
{answer}

A resposta está COMPLETA e SUFICIENTE para responder a pergunta?

Responda apenas em JSON:

{{
  "sufficient": true ou false
}}
"""

        response = await self._llm_call(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("sufficient", True)
        except:
            return True

    # =====================================
    # AGENT LOOP
    # =====================================
    @mirror_trace("chat_request")
    @observe("chat_request")
    @trace_async("chat_request")
    async def chat(
        self,
        message: str,
        collection: str = None,
        system_prompt: str = None,
        sessionData: dict = None,
    ) -> APIResponse:


        session_id = start_trace_session() 

        _mirror_tracer.log_custom_event("user.input", {
            "question": message
        })
        
        candidate_answer = None
        try:
            with span("chat.start", question=message):
                question = message

                if not collection and sessionData:
                    collection_info = sessionData.get("collectionINFO") or {}
                    collection = collection_info.get("name") or os.getenv("QDRANT_COLLECTION")

                system_prompt = system_prompt or (
                    "Você é um assistente da empresa Servopa.\n\n"
                    "FERRAMENTAS DISPONÍVEIS:\n\n"
                    "1. 'open_canvas' — Use quando o usuário pedir para CRIAR, ESCREVER, GERAR, ELABORAR, "
                    "DOCUMENTAR ou PRODUZIR qualquer conteúdo longo. "
                    "Exemplos: 'escreva sobre X', 'abre um canvas', 'crie um texto sobre X'.\n\n"
                    "2. 'buscar_qdrant' — Use quando a pergunta for sobre Servopa, consórcio, "
                    "financiamento, veículos ou serviços da empresa.\n\n"
                    "3. COMBINAÇÃO OBRIGATÓRIA: Se o usuário pedir para ESCREVER/CRIAR algo sobre a EMPRESA "
                    "(Servopa, veículos, consórcio, financiamento), chame 'open_canvas' E 'buscar_qdrant' "
                    "AO MESMO TEMPO para enriquecer o canvas com dados reais da base de conhecimento.\n\n"
                    "REGRA OBRIGATÓRIA: SEMPRE chame ao menos uma ferramenta. NUNCA responda diretamente.\n\n"
                    "Nunca invente informações. Responda com base no contexto fornecido pelas ferramentas."
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]

                trace = []
                tools_used = []

                MAX_ITERATIONS = 5

                
                for i in range(MAX_ITERATIONS):

                    with span("agent.iteration", iteration=i):

                        trace.append(f"Iteração {i+1}: analisando pergunta")

                        with span("llm.call", model=self._deployment):
                            response = await self._llm_call(
                                model=self._deployment,
                                messages=messages,
                                tools=_TOOLS,
                                tool_choice="auto"
                            )

                            msg = response.choices[0].message

                            logger.info(f"Iteration {i}: {msg}")
                    

                    result = ""
                    # =================================
                    # TOOL CALL
                    # =================================

                    if msg.tool_calls:

                        tool_call = msg.tool_calls[0]
                        tool_name = tool_call.function.name


                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except:
                            arguments = {"question": question}

                        _mirror_tracer.log_custom_event("tool.call", {
                            "tool_name": tool_name,
                            "arguments": arguments
                        })

                        trace.append(f"Chamando ferramenta: {tool_name}")
                        tools_used.append(tool_name)
                        
                        with span("tool.call", tool_name=tool_name):
                        
                            # Inicializa result para todas as tools
                            result = ""
                        
                            if tool_name == "buscar_qdrant":

                                if not collection:
                                    return APIResponse(
                                        status=False,
                                        message="Nenhuma coleção Qdrant definida",
                                        data=None
                                    )

                                result = await self._buscar_qdrant(arguments["question"], collection)
                                
                                _mirror_tracer.log_custom_event("tool.result", {
                                    "tool_name": tool_name,
                                    "result_preview": str(result)[:500]
                                })
                                
                            elif tool_name == "open_canvas":
                                # In HTTP mode canvas streaming is not available;
                                # generate the content and return it directly.
                                canvas_title = arguments.get("title", "Canvas")
                                prompt = arguments.get("prompt", question)
                                canvas_resp = await self._llm_call(
                                    model=self._deployment,
                                    messages=[
                                        {"role": "system", "content": "Você é um escritor especialista. Escreva de forma clara, detalhada e bem estruturada em Markdown."},
                                        {"role": "user", "content": prompt},
                                    ],
                                )
                                return APIResponse(
                                    status=True,
                                    message="OK",
                                    data={
                                        "response": canvas_resp.choices[0].message.content,
                                        "tool_used": "open_canvas",
                                        "canvas_title": canvas_title,
                                    }
                                )
                            
                            else:
                                # Tool desconhecida - retorna erro
                                result = f"Tool '{tool_name}' não é reconhecida"

                    

                        messages.append(msg)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })

                        continue

                      
                # =================================
                # CANDIDATE ANSWER
                # =================================

                    candidate_answer = msg.content

                    trace.append("Resposta candidata gerada")
                    with span("llm.evaluation"):
                        sufficient = await self._avaliar_resposta(
                            question,
                            candidate_answer
                        )

                    if sufficient:

                        trace.append("Resposta considerada suficiente")

                        return APIResponse(
                            status=True,
                            message="OK",
                            data={
                                "response": candidate_answer,
                                "trace": trace,
                                "tools_used": list(set(tools_used))
                            }
                            
                        )
                    
                    else:

                        trace.append("Resposta considerada incompleta, continuando busca")

                        messages.append({
                            "role": "assistant",
                            "content": "A resposta anterior pode estar incompleta. Continue investigando."
                        })

                        continue

            return APIResponse(
                status=False,
                message="Limite de iterações do agente atingido",
                data={"trace": trace}
            )

        except Exception as e:

            logger.error(f"ChatClient.chat error: {e}", exc_info=True)

            _mirror_tracer.log_custom_event("assistant.output", {
                "response": candidate_answer
            })
            return APIResponse(
                status=False,
                message=str(e),
                data=None
            )

    # =====================================
    # STREAMING (WebSocket)
    # =====================================
    @mirror_trace("chat_stream")
    @observe("chat_stream")
    @trace_async_generator("chat_stream")
    async def chat_stream(
        self,
        message: str,
        collection: str = None,
        sessionData: dict = None,
    ):
        from uuid import uuid4
        step_id = str(uuid4())[:8]

        try:
            question = message
            logger.info(f"[chat_stream] step_id={step_id} | question={question!r}")

            if not collection and sessionData:
                collection_info = sessionData.get("collectionINFO") or {}
                collection = collection_info.get("name") or os.getenv("QDRANT_COLLECTION", "").strip()

            system_prompt = (
                "Você é um assistente da empresa Servopa.\n\n"
                "FERRAMENTAS DISPONÍVEIS:\n\n"
                "1. 'open_canvas' — Use quando o usuário pedir para CRIAR, ESCREVER, GERAR, ELABORAR, "
                "DOCUMENTAR ou PRODUZIR qualquer conteúdo longo. "
                "Exemplos: 'escreva sobre X', 'abre um canvas', 'crie um texto sobre X'.\n\n"
                "2. 'buscar_qdrant' — Use quando a pergunta for sobre Servopa, consórcio, "
                "financiamento, veículos ou serviços da empresa.\n\n"
                "3. 'open_stats_widget' — Use SOMENTE para Qdrant ou Redis. Valores: "
                "qdrant_total, qdrant_enabled, qdrant_disabled, redis_total, redis_enabled, redis_disabled. "
                "Para 'chunks/registros ativos no total' chame DUAS VEZES (qdrant_enabled + redis_enabled). "
                "Se o banco citado não for Qdrant nem Redis, use 'answer'.\n\n"
                "4. 'answer' — Use para responder diretamente: saudações, perguntas sobre suas "
                "capacidades, pedidos inválidos, ambíguos ou fora do escopo. NUNCA retorne texto "
                "livre — sempre chame esta ferramenta para textos diretos.\n\n"
                "5. COMBINAÇÃO OBRIGATÓRIA: Se o usuário pedir para ESCREVER/CRIAR algo sobre a EMPRESA "
                "(Servopa, veículos, consórcio, financiamento), chame 'open_canvas' E 'buscar_qdrant' "
                "AO MESMO TEMPO para enriquecer o canvas com dados reais da base de conhecimento.\n\n"
                "REGRA OBRIGATÓRIA: SEMPRE chame uma ferramenta. NUNCA responda como texto livre."
            )

            # Non-streaming first call to get tool selection
            response = await self._llm_call(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                tools=_TOOLS,
                tool_choice="required",
            )

            msg = response.choices[0].message
            tool_used = "direct"
            logger.info(f"[chat_stream] Iteration 0: {msg}")

            if msg.tool_calls:
                all_tools: dict[str, dict] = {}
                all_tool_calls: list[tuple[str, dict]] = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    all_tool_calls.append((tc.function.name, args))
                    all_tools[tc.function.name] = args  # mantém para checks de presença

                tool_used = "+".join(all_tools.keys())
                logger.info(f"[chat_stream] tools selected: {list(all_tools.keys())} | args: {all_tools}")

                for name in all_tools:
                    yield {"type": "function_call", "step_id": step_id, "content": name}

                if "answer" in all_tools:
                    answer_text = all_tools["answer"].get("text", "")

                    _mirror_tracer.log_custom_event("llm.response", {
                        "response": answer_text,
                        "source": "tool_answer"
                    })    

                    if answer_text:
                        yield {"type": "text_delta", "step_id": step_id, "content": answer_text}
                    yield {"type": "done", "step_id": step_id, "tool_used": "answer"}
                    return

                stats_calls = [(n, a) for n, a in all_tool_calls if n == "open_stats_widget"]
                if stats_calls and "open_canvas" not in all_tools:
                    stats_id = f"stats-{str(uuid4())[:8]}"
                    widgets = [a.get("widget", "qdrant_enabled") for _, a in stats_calls]
                    labels = ", ".join([_WIDGET_LABELS.get(w, w) for w in widgets])

                    # Gera texto contextual PRIMEIRO — a bolha de texto aparece antes do widget
                    text_stream = await self._llm_call(
                        model=self._deployment,
                        messages=[
                            {"role": "system", "content": "Você é um assistente conciso."},
                            {
                                "role": "user",
                                "content": (
                                    f"O usuário perguntou: '{question}'\n"
                                    f"Indicadores exibidos no chat: {labels}.\n"
                                    "Responda em uma frase direta informando o que está sendo exibido. "
                                    "Não mencione valores numéricos — eles já aparecem no card."
                                ),
                            },
                        ],
                        stream=True,
                    )
                    async for chunk in text_stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield {"type": "text_delta", "step_id": step_id, "content": delta}

                    # Depois do texto, emite o widget
                    yield {
                        "type": "component_open",
                        "step_id": stats_id,
                        "component": "stats_widget",
                        "widgets": widgets,
                        "parent_step_id": step_id,
                        "tool_used": tool_used,
                    }

                    yield {"type": "done", "step_id": stats_id, "tool_used": tool_used}
                    yield {"type": "done", "step_id": step_id, "tool_used": tool_used}
                    return

                if "open_canvas" in all_tools:
                    canvas_args = all_tools["open_canvas"]
                    canvas_id = f"canvas-{str(uuid4())[:8]}"
                    title = canvas_args.get("title", "Canvas")
                    prompt = canvas_args.get("prompt", question)

                    yield {
                        "type": "component_open",
                        "step_id": canvas_id,
                        "component": "canvas",
                        "title": title,
                        "parent_step_id": step_id,
                        "tool_used": tool_used,
                    }

                    # If buscar_qdrant was also called, fetch context to ground the canvas
                    context = ""
                    if "buscar_qdrant" in all_tools and collection:
                        qdrant_q = all_tools["buscar_qdrant"].get("question", question)
                        logger.info(f"[chat_stream] fetching qdrant context for canvas | q={qdrant_q!r}")
                        context = await self._buscar_qdrant(qdrant_q, collection)

                    canvas_system = (
                        "Você é um escritor especialista. Escreva de forma clara, detalhada e bem "
                        "estruturada em Markdown."
                        + (" Use o contexto da base de conhecimento como base factual — não invente dados." if context else "")
                    )
                    canvas_user = (
                        f"Contexto da base de conhecimento:\n{context}\n\nInstrução: {prompt}"
                        if context else prompt
                    )

                    canvas_stream = await self._llm_call(
                        model=self._deployment,
                        messages=[
                            {"role": "system", "content": canvas_system},
                            {"role": "user", "content": canvas_user},
                        ],
                        stream=True,
                    )
                    async for chunk in canvas_stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield {"type": "text_delta", "step_id": canvas_id, "content": delta}

                    yield {"type": "done", "step_id": canvas_id, "tool_used": tool_used}
                    yield {"type": "done", "step_id": step_id, "tool_used": tool_used}
                    return

                # ── buscar_qdrant only (no canvas) ──
                elif "buscar_qdrant" in all_tools:
                    if not collection:
                        yield {"type": "error", "step_id": step_id, "content": "Nenhuma coleção especificada."}
                        return

                    qdrant_q = all_tools["buscar_qdrant"].get("question", question)
                    context = await self._buscar_qdrant(qdrant_q, collection)
                    messages_final = [
                        {
                            "role": "system",
                            "content": (
                                "Você é um assistente útil. "
                                "Responda utilizando apenas o contexto fornecido. "
                                "Se a informação não estiver no contexto, diga: "
                                "'Não encontrei essa informação na base de conhecimento.'"
                            ),
                        },
                        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta:\n{question}"},
                    ]

                else:
                    # Unknown / other tool
                    first_args = next(iter(all_tools.values()))
                    messages_final = [
                        {"role": "system", "content": "Responda a pergunta do usuário."},
                        {"role": "user", "content": first_args.get("question", question)},
                    ]

                stream = await self._llm_call(
                    model=self._deployment,
                    messages=messages_final,
                    stream=True,
                )

            else:
                # Model answered directly — stream that response
                stream = await self._llm_call(
                    model=self._deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    stream=True,
                )
            full_response = ""
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta 
                    yield {"type": "text_delta", "step_id": step_id, "content": delta}
                _mirror_tracer.log_custom_event("llm.response", {
                    "response": full_response
                })
            yield {"type": "done", "step_id": step_id, "tool_used": tool_used}

        except Exception as e:
            logger.error(f"ChatClient.chat_stream error: {e}", exc_info=True)
            yield {"type": "error", "step_id": step_id, "content": str(e)}