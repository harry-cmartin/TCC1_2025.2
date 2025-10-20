# TCC1_2025.2

## Tecnologias

Python
Cypher
Neo4j
Docker
ChatGPT 4.0+ versions
Claude Sonnet

## Como gerar o grafo

Como a proposta do Trabalho é Engenharia de prompt, usaremos o Copilot com os modelos Claude Sonnet para geração dos grafos.

## Fontes dos dados

- **Zenodo**: Utilizamos um csv feito com o propósito de treinar redes neura, e contem centenas de linhas com as histórias de usuário (user_stories). O csv é disponibilizado no site Zenodo.

## Modelo pra embeddings

Para representar semanticamente os nós e relacionamentos do grafo, utilizaremos embeddings de texto:

- **OpenAI Embeddings**: API da OpenAI para embeddings de alta qualidade, integrável via prompts para análise de requisitos.

## Como sera armazenado no Neo4j

Os dados serão armazenados no Neo4j como um grafo, com:

- **Nós**: Representando entidades como Requirement, Technique, Instruction e Concept, com propriedades (ex.: name, description, domain).
- **Relacionamentos**: Conectando nós com tipos específicos (ex.: USES, APPLIES_TO, RELATES_TO), também com propriedades opcionais.
- **Persistência**: Dados inseridos via queries Cypher no script Python, permitindo consultas e visualização no Neo4j Browser.
- **Integração com Embeddings**: Vetores de embeddings armazenados como arrays em propriedades dos nós para buscas semânticas.

![Especificacao](TCC1_2025.2/Especificacao_do_grafo.md)
