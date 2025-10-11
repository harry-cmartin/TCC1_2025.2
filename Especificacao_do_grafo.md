Perfeito 🔥 — entendi tua visão direitinho:


# 🧩 **Modelo de Grafo de Conhecimento para Engenharia de Requisitos (com embeddings)**

## 🌐 Tipos de Nós

### 1. **Requirement** — Requisitos de software (instâncias reais)

Representa um requisito funcional ou não funcional de algum sistema.

```plaintext
Label: :Requirement
```

**Propriedades:**

| Propriedade       | Tipo     | Descrição                                          |
| ----------------- | -------- | -------------------------------------------------- |
| `req_id`          | string   | Identificador único                                |
| `text`            | string   | Texto do requisito                                 |
| `summary`         | string   | Resumo curto                                       |
| `type`            | string   | "funcional", "não-funcional", "user-story"         |
| `source`          | string   | Origem (ex: dataset, projeto, sistema)             |
| `domain`          | string   | Domínio do software (financeiro, e-commerce, etc.) |
| `embedding`       | float[]  | Vetor de embedding do texto                        |
| `embedding_model` | string   | Modelo usado para gerar embedding                  |
| `embedding_ts`    | datetime | Data/hora da geração                               |
| `created_at`      | datetime | Data de inclusão                                   |

---

### 2. **Technique** — Técnicas, métodos e práticas da Engenharia de Requisitos

Representa técnicas como *entrevistas, prototipação, análise de stakeholders, casos de uso, elicitação etc.*

```plaintext
Label: :Technique
```

**Propriedades:**

| Propriedade       | Tipo     | Descrição                                  |
| ----------------- | -------- | ------------------------------------------ |
| `tech_id`         | string   | Identificador único                        |
| `name`            | string   | Nome da técnica                            |
| `description`     | string   | Descrição resumida                         |
| `category`        | string   | Elicitação, Especificação, Validação, etc. |
| `source`          | string   | Livro/artigo de origem                     |
| `embedding`       | float[]  | Vetor semântico do texto descritivo        |
| `embedding_model` | string   | Modelo de embedding usado                  |
| `embedding_ts`    | datetime | Data/hora da geração                       |
| `created_at`      | datetime | Data de inclusão                           |

---

### 3. **Instruction** — Diretrizes e boas práticas extraídas de livros/artigos

São frases ou recomendações que descrevem *como agir* no processo de engenharia de requisitos.
Exemplo: “Os requisitos devem ser verificáveis e testáveis.”

```plaintext
Label: :Instruction
```

**Propriedades:**

| Propriedade       | Tipo     | Descrição                                        |
| ----------------- | -------- | ------------------------------------------------ |
| `instr_id`        | string   | Identificador único                              |
| `text`            | string   | Texto da instrução                               |
| `context`         | string   | Contexto (por ex.: "Especificação", "Validação") |
| `source`          | string   | Livro ou autor de origem                         |
| `embedding`       | float[]  | Embedding do texto                               |
| `embedding_model` | string   | Modelo de embedding usado                        |
| `embedding_ts`    | datetime | Data/hora da geração                             |
| `created_at`      | datetime | Data de inclusão                                 |

---

### 4. **Concept** — Conceitos gerais de Engenharia de Software

Usado para conectar entidades abstratas (ex: “Stakeholder”, “Requisito Funcional”, “Prototipação”, “Documento de Especificação”)

```plaintext
Label: :Concept
```

**Propriedades:**

| Propriedade       | Tipo     | Descrição             |
| ----------------- | -------- | --------------------- |
| `concept_id`      | string   | Identificador         |
| `name`            | string   | Nome do conceito      |
| `definition`      | string   | Definição resumida    |
| `source`          | string   | Fonte teórica         |
| `embedding`       | float[]  | Embedding do conceito |
| `embedding_model` | string   | Modelo usado          |
| `embedding_ts`    | datetime | Data/hora             |
| `created_at`      | datetime | Data de inclusão      |

---

## 🔗 **Relações Principais**

| Relação                                              | Direção                   | Descrição                                                  | Exemplo                                                                             |
| ---------------------------------------------------- | ------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `(:Requirement)-[:USES_TECHNIQUE]->(:Technique)`     | Requirement → Technique   | Técnica usada para obter ou especificar aquele requisito   | “Requisito de login” usa “Entrevistas com usuários”                                 |
| `(:Instruction)-[:REFERS_TO]->(:Concept)`            | Instruction → Concept     | A instrução refere-se a um conceito teórico                | “Requisitos devem ser verificáveis” → “Requisito”                                   |
| `(:Requirement)-[:IS_RELATED_TO]->(:Concept)`        | Requirement → Concept     | Conexão semântica entre um requisito e um conceito técnico | “Login” → “Autenticação”                                                            |
| `(:Technique)-[:APPLIES_TO]->(:Concept)`             | Technique → Concept       | Técnica aplicável a um conceito específico                 | “Casos de uso” → “Requisitos funcionais”                                            |
| `(:Instruction)-[:SUGGESTS_TECHNIQUE]->(:Technique)` | Instruction → Technique   | A instrução sugere usar determinada técnica                | “Entrevistar stakeholders” → “Entrevistas”                                          |
| `(:Requirement)-[:SUPPORTED_BY]->(:Instruction)`     | Requirement → Instruction | O requisito segue uma instrução/boa prática teórica        | “O sistema deve permitir autenticação segura” → “Requisitos devem ser verificáveis” |

---

## 💾 **Exemplo de criação no Neo4j (Cypher com embeddings dentro dos nós)**

```cypher
// Exemplo de requisito real
CREATE (r:Requirement {
    req_id: "REQ001",
    text: "O sistema deve permitir que o usuário redefina a senha via e-mail.",
    type: "funcional",
    domain: "segurança",
    source: "ProjetoX",
    embedding: [0.123, -0.456, 0.789, ...],
    embedding_model: "text-embedding-3-small",
    embedding_ts: datetime(),
    created_at: datetime()
})

// Técnica teórica
CREATE (t:Technique {
    tech_id: "TECH001",
    name: "Entrevistas",
    description: "Coleta de requisitos por meio de entrevistas com stakeholders.",
    category: "Elicitação",
    source: "Sommerville (2011)",
    embedding: [0.223, -0.316, 0.444, ...],
    embedding_model: "text-embedding-3-small",
    embedding_ts: datetime(),
    created_at: datetime()
})

// Instrução de livro
CREATE (i:Instruction {
    instr_id: "INST001",
    text: "Os requisitos devem ser claros e verificáveis.",
    context: "Especificação",
    source: "Sommerville (2011)",
    embedding: [0.512, -0.122, 0.211, ...],
    embedding_model: "text-embedding-3-small",
    embedding_ts: datetime(),
    created_at: datetime()
})

// Conceito teórico
CREATE (c:Concept {
    concept_id: "C001",
    name: "Requisito Funcional",
    definition: "Um requisito que descreve uma função específica do sistema.",
    source: "IEEE Std 830",
    embedding: [0.311, -0.244, 0.665, ...],
    embedding_model: "text-embedding-3-small",
    embedding_ts: datetime(),
    created_at: datetime()
})

// Relações
CREATE (r)-[:IS_RELATED_TO]->(c)
CREATE (i)-[:REFERS_TO]->(c)
CREATE (i)-[:SUGGESTS_TECHNIQUE]->(t)
```


