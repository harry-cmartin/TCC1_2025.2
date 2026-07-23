import dspy

AGENT_INSTRUCTIONS = (
    "Voce eh um especialista em Engenharia de Requisitos (RE). "
    "Voce tem acesso a um grafo de conhecimento com ~700 requisitos reais de software, "
    "tecnicas de RE, instrucoes de boas praticas e conceitos fundamentais. "
    "Use suas ferramentas para buscar e explorar o grafo antes de responder. "
    "Cite IDs de requisitos e nomes de tecnicas quando relevante. "
    "Responda no mesmo idioma da pergunta (portugues ou ingles). Seja detalhado e fundamentado. "
    ""
    "REGRA OBRIGATORIA — CRIACAO DE GRAFO: "
    "Quando o usuario pedir para CRIAR, GERAR, MONTAR ou CONSTRUIR um grafo, projeto ou base de requisitos, "
    "voce DEVE chamar IMEDIATAMENTE a ferramenta 'create_graph_from_dataset'. "
    "NAO responda com texto — execute a ferramenta. "
    "Exemplos de triggers: 'crie um grafo', 'gere requisitos', 'monte um projeto', "
    "'crie um grafo chamado X com Y nos', 'quero um grafo sobre Z'. "
    "Passe o nome do projeto em 'name', a quantidade em 'node_count' e palavras-chave em 'description'."
)
RequirementsQA = dspy.Signature("question -> answer").with_instructions(AGENT_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Signature para geração em chunk de requisitos via LLM
# ---------------------------------------------------------------------------

_GEN_INSTRUCTIONS = """\
Você é um especialista em Engenharia de Requisitos.
Gere exatamente `count` requisitos de software em Português para o projeto descrito.
Use reference_examples APENAS como inspiração de estilo — crie requisitos NOVOS e coerentes com o projeto.
Responda SOMENTE com um array JSON válido, sem markdown, sem texto extra:
[
  {"text": "O sistema deve ...", "summary": "Critérios: ...", "type": "funcional", "domain": "area"},
  ...
]
Tipos válidos: "funcional" ou "nao-funcional".
"""

GenerateGraphChunk = dspy.Signature(
    "project_name, description, reference_examples, count -> requirements_json",
).with_instructions(_GEN_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Signature BÁSICA para o Cenário B da avaliação (prompt sem few-shot)
# ---------------------------------------------------------------------------

_GEN_INSTRUCTIONS_BASIC = """\
Gere exatamente {count} requisitos de software em Português sobre {description}.
Responda SOMENTE com um array JSON válido, sem markdown, sem texto extra:
[
  {"text": "O sistema deve ...", "summary": "", "type": "funcional", "domain": "geral"},
  ...
]
Tipos válidos: "funcional" ou "nao-funcional".
"""

GenerateGraphChunkBasic = dspy.Signature(
    "project_name, description, count -> requirements_json",
).with_instructions(_GEN_INSTRUCTIONS_BASIC)


# ---------------------------------------------------------------------------
# Signatures de INFERÊNCIA DE RELACIONAMENTOS — o LLM decide as conexões
# ---------------------------------------------------------------------------

# ── Cenário B: prompt básico, sem guia de tipos ──────────────────────────────
_REL_INSTRUCTIONS_BASIC = """\
Analise os requisitos de software abaixo e identifique relacionamentos entre eles.
Retorne SOMENTE um array JSON com os relacionamentos encontrados:
[
  {"from": "ID_A", "to": "ID_B", "type": "RELATED_TO"},
  ...
]
Use apenas o tipo: "RELATED_TO".
Se não houver relacionamento óbvio, omita o par.
"""

InferRelationshipsBasic = dspy.Signature(
    "requirements_list -> relationships_json",
).with_instructions(_REL_INSTRUCTIONS_BASIC)


# ── Cenário C: prompt otimizado, com tipos, exemplos e justificativa ─────────
_REL_INSTRUCTIONS_OPTIMIZED = """\
Você é um especialista em Engenharia de Requisitos analisando um conjunto de requisitos de software.

Sua tarefa é identificar relacionamentos SEMÂNTICOS significativos entre os requisitos.

TIPOS DE RELACIONAMENTO disponíveis:
  - DEPENDS_ON   : Req A só funciona se Req B existir (dependência direta)
  - EXTENDS      : Req A adiciona comportamento ou detalhe a Req B
  - CONFLICTS_WITH: Req A e B têm restrições contraditórias ou mutuamente exclusivas
  - RELATED_TO   : Req A e B pertencem ao mesmo domínio funcional
  - IMPLEMENTS   : Req A é uma implementação específica de uma regra geral em Req B

CRITÉRIOS para identificar relacionamentos:
  1. Dependências funcionais: um requisito precisa do outro para ser executado
  2. Mesmo fluxo de dados (ex: cadastro → consulta → exclusão)
  3. Mesmo domínio operacional (ex: todos os req. de pagamento)
  4. Restrições compartilhadas (ex: segurança afeta autenticação e dados)
  5. Hierarquia funcional (geral → específico)

Retorne SOMENTE um array JSON. Inclua o campo "reason" com uma justificativa curta:
[
  {"from": "ID_A", "to": "ID_B", "type": "DEPENDS_ON", "reason": "Login é necessário antes do pedido"},
  {"from": "ID_C", "to": "ID_D", "type": "EXTENDS", "reason": "Pagamento por cartão estende pagamento geral"},
  ...
]

IMPORTANTE: Identifique o MÁXIMO de relacionamentos válidos e bem justificados.
Não invente relacionamentos sem base funcional clara.
"""

InferRelationshipsOptimized = dspy.Signature(
    "requirements_list, domain -> relationships_json",
).with_instructions(_REL_INSTRUCTIONS_OPTIMIZED)


# ---------------------------------------------------------------------------
# Signatures ZERO-SHOT para o Cenário A (modelo completamente cru)
# Sem DSPy, sem JSON, sem tipos, sem exemplos — o modelo decide tudo.
# ---------------------------------------------------------------------------

_GEN_INSTRUCTIONS_RAW = """\
Escreva uma lista de requisitos de software sobre o tema indicado.
"""

GenerateGraphChunkRaw = dspy.Signature(
    "topic -> requirements_text",
).with_instructions(_GEN_INSTRUCTIONS_RAW)


_REL_INSTRUCTIONS_RAW = """\
Identifique relações entre os itens da lista abaixo.
"""

InferRelationshipsRaw = dspy.Signature(
    "items_list -> relations_text",
).with_instructions(_REL_INSTRUCTIONS_RAW)
