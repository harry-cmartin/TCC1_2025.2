# Grafo de Conhecimento de Engenharia de Requisitos

Este projeto implementa um grafo de conhecimento para Engenharia de Requisitos usando Neo4j e Python, baseado na especificação do arquivo `Especificacao_do_grafo.md`.

## 🏗️ Estrutura do Grafo

O grafo possui 4 tipos de nós:
- **Requirement**: Requisitos de software (funcionais/não-funcionais)
- **Technique**: Técnicas de engenharia de requisitos
- **Instruction**: Diretrizes e boas práticas
- **Concept**: Conceitos gerais de engenharia de software

## 🚀 Como executar

### 1. Pré-requisitos
- Docker instalado
- Python 3.7+

### 2. Subir o Neo4j no Docker
```bash
docker run -d \
  --name neo4j \
  -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword \
  -v $HOME/neo4j/data:/data \
  neo4j:latest
```

### 3. Instalar dependências Python
```bash
pip install -r requirements.txt
```

### 4. Executar o script
```bash
python graph_creator.py
```

## 🌐 Visualização

Após executar o script, acesse:
- **Neo4j Browser**: http://localhost:7474
- **Credenciais**: neo4j / testpassword

### Queries úteis para visualização:

```cypher
// Visualizar todo o grafo
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25

// Contar nós por tipo
MATCH (n)
RETURN labels(n)[0] as tipo, count(n) as total
ORDER BY tipo

// Ver todos os requisitos e suas conexões
MATCH (r:Requirement)-[rel]->(other)
RETURN r, rel, other

// Buscar por domínio específico
MATCH (r:Requirement {domain: "segurança"})
RETURN r

// Ver técnicas de elicitação
MATCH (t:Technique {category: "Elicitação"})
RETURN t
```

## 📂 Arquivos

- `graph_creator.py`: Script principal para criar o grafo
- `requirements.txt`: Dependências Python
- `.env`: Configurações de conexão com Neo4j
- `Especificacao_do_grafo.md`: Especificação completa do modelo

## 🔧 Funcionalidades

O script `graph_creator.py` oferece:
- Conexão automática com Neo4j usando variáveis de ambiente
- Criação de todos os tipos de nós com propriedades completas
- Criação de todos os relacionamentos especificados
- População com dados de exemplo
- Estatísticas do grafo criado
- Context manager para gerenciar conexões

## 🎯 Dados de Exemplo

O script cria automaticamente:
- 3 Requirements de exemplo
- 3 Techniques diferentes
- 3 Instructions de boas práticas
- 3 Concepts fundamentais
- Relacionamentos conectando todos os elementos

## 📊 Estatísticas

Após a execução, você verá:
- Contagem de nós por tipo
- Contagem de relacionamentos por tipo
- Total geral de elementos no grafo