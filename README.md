# TCC1_2025.2 - Grafo de Conhecimento em Engenharia de Requisitos

## Visão Geral do Projeto

Este projeto implementa um **grafo de conhecimento** para Engenharia de Requisitos, utilizando tecnologias modernas como Neo4j, Python e IA generativa. O objetivo é modelar relacionamentos entre requisitos de software, técnicas de elicitação, diretrizes e conceitos fundamentais, facilitando a análise e descoberta de conhecimento na área.

O projeto é parte do TCC (Trabalho de Conclusão de Curso) de 2025.2, com foco em **Engenharia de Prompt** para geração automática de grafos usando modelos de IA como Claude Sonnet.

## Tecnologias Utilizadas

- **Python 3.7+**: Linguagem principal para desenvolvimento
- **Neo4j**: Banco de dados de grafos para armazenamento e consultas
- **Cypher**: Linguagem de consulta do Neo4j
- **Docker**: Containerização do Neo4j para desenvolvimento
- **ChatGPT 5.1+**: IA para geração de prompts e conteúdo
- **Claude Sonnet**: Modelo principal para engenharia de prompt e geração de grafos
- **python-dotenv**: Gerenciamento de variáveis de ambiente
- **neo4j-driver**: Driver oficial para conexão Python-Neo4j

## Estrutura do Grafo

O grafo é composto por 4 tipos principais de nós:

### Nós (Nodes)

- **Requirement**: Requisitos funcionais/não-funcionais
  - Propriedades: `req_id`, `text`, `summary`, `type`, `source`, `domain`, `embedding`, etc.
- **Technique**: Técnicas de engenharia de requisitos (elicitação, especificação, validação)
  - Propriedades: `tech_id`, `name`, `description`, `category`, `source`, `embedding`, etc.
- **Instruction**: Diretrizes e boas práticas
  - Propriedades: `instr_id`, `text`, `context`, `source`, `embedding`, etc.
- **Concept**: Conceitos fundamentais
  - Propriedades: `concept_id`, `name`, `definition`, `source`, `embedding`, etc.

### Relacionamentos (Relationships)

- `USES_TECHNIQUE`: Requirement → Technique
- `REFERS_TO`: Instruction → Concept
- `IS_RELATED_TO`: Requirement → Concept
- `APPLIES_TO`: Technique → Concept
- `SUGGESTS_TECHNIQUE`: Instruction → Technique
- `SUPPORTED_BY`: Requirement → Instruction

## Como Executar

### Pré-requisitos

- Docker instalado
- Python 3.7+
- Conta no Neo4j (opcional, pode usar local)

### 1. Subir Neo4j via Docker

```bash
docker run -d \
  --name neo4j \
  -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword \
  -v $HOME/neo4j/data:/data \
  neo4j:latest
```

### 2. Configurar Ambiente

```bash
cd Knowledge_Graphs
pip install -r ../requirements.txt
```

### 3. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na pasta `Knowledge_Graphs`:

```env
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword
NEO4J_DATABASE=neo4j
```

### 4. Executar o Script

```bash
python graph_creator.py
```

### 5. Visualizar

Após executar o script, acesse:
- **Neo4j Browser**: http://localhost:7474
- **Credenciais**: neo4j / testpassword

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 700
```

##  Explicação Detalhada do Código

### graph_creator.py

O script principal é organizado em uma classe `Neo4jGraphCreator` que gerencia a criação e população do grafo.

#### Classe Neo4jGraphCreator

**Inicialização (`__init__`)**:

- Carrega variáveis de ambiente do arquivo `.env`
- Configura URI, usuário, senha e banco de dados
- Inicializa o driver como `None`

**Conexão (`connect`)**:

- Cria instância do driver Neo4j
- Testa conexão executando query simples
- Retorna `True` se sucesso, `False` caso contrário

**Context Manager**:

- `__enter__`: Chama `connect()` e retorna a instância
- `__exit__`: Chama `close()` para fechar conexão

#### Métodos de Criação de Nós

Cada tipo de nó tem seu método específico:

**`create_requirement`**:

- Cria nós do tipo `:Requirement`
- Parâmetros: `req_id`, `text`, `summary`, `req_type`, `source`, `domain`, `embedding`
- Query Cypher cria nó com todas as propriedades, incluindo timestamp de criação

**`create_technique`**:

- Similar ao requirement, mas para técnicas
- Propriedades: `tech_id`, `name`, `description`, `category`, `source`, `embedding`

**`create_instruction`**:

- Para diretrizes e instruções
- Propriedades: `instr_id`, `text`, `context`, `source`, `embedding`

**`create_concept`**:

- Para conceitos fundamentais
- Propriedades: `concept_id`, `name`, `definition`, `source`, `embedding`

#### População de Dados (`populate_sample_data`)

Este método cria dados de exemplo baseados na especificação:

1. **Cria 3 Concepts**: Requisito Funcional, Autenticação, Stakeholder
2. **Cria 3 Techniques**: Entrevistas, Casos de Uso, Prototipação
3. **Cria 3 Instructions**: Requisitos verificáveis, entrevistar stakeholders, validar com protótipos
4. **Cria 3 Requirements**: Redefinição de senha, autenticação básica, performance
5. **Cria relacionamentos** conectando todos os nós

Os embeddings são vetores de exemplo (hardcoded) para demonstração.

#### Estatísticas (`get_graph_statistics`)

Executa queries Cypher para contar:

- Nós por tipo de label
- Relacionamentos por tipo
- Totais gerais

#### Função Main

Orquestra todo o processo:

1. Conecta ao Neo4j
2. Limpa o banco (atenção em produção!)
3. Popula com dados de exemplo
4. Mostra estatísticas
5. Fornece instruções para visualização

## Modelo de Embeddings

Para representação semântica, utilizamos embeddings de texto.

## Armazenamento no Neo4j

### Estrutura de Dados

**Nós**:

- Labels: `:Requirement`, `:Technique`, `:Instruction`, `:Concept`
- Propriedades comuns: `id`, `name/text`, `description`, `source`, `embedding`, `created_at`


### Persistência

- Dados inseridos via queries Cypher no Python
- Consultas e visualização no Neo4j Browser
- Suporte a buscas semânticas via embeddings



