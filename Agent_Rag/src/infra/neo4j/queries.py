"""
Queries Cypher isoladas — todas as strings ficam aqui.
"""

SEARCH_REQUIREMENTS = """
MATCH (r:Requirement)
WHERE ($graph_id = '' OR r.graph_id = $graph_id)
  AND (
    any(kw IN $keywords WHERE toLower(r.text) CONTAINS kw)
    OR any(kw IN $keywords WHERE toLower(r.summary) CONTAINS kw)
    OR any(kw IN $keywords WHERE toLower(coalesce(r.domain,'')) CONTAINS kw)
  )
WITH r,
     size([kw IN $keywords WHERE toLower(r.text) CONTAINS kw]) * 3 +
     size([kw IN $keywords WHERE toLower(r.summary) CONTAINS kw]) * 2 +
     size([kw IN $keywords WHERE toLower(coalesce(r.domain,'')) CONTAINS kw]) AS score
ORDER BY score DESC
LIMIT $limit
RETURN r.req_id AS req_id, r.text AS text, r.summary AS summary,
       r.type AS type, r.domain AS domain, r.communityId AS communityId, score
"""

GET_REQUIREMENT_CONTEXT = """
MATCH (r:Requirement {req_id: $req_id})
OPTIONAL MATCH (r)-[:USES_TECHNIQUE]->(t:Technique)
OPTIONAL MATCH (r)-[:IS_A]->(c1:Concept)
OPTIONAL MATCH (r)-[:SUPPORTED_BY]->(i:Instruction)
// Vizinhos semânticos: todas as arestas criadas pelo LLM
OPTIONAL MATCH (r)-[rel:RELATED_TO|DEPENDS_ON|EXTENDS|CONFLICTS_WITH|IMPLEMENTS|SIMILAR_TO]->(nb:Requirement)
WITH r,
     collect(DISTINCT t.name) AS techniques,
     collect(DISTINCT c1.name) AS concepts,
     collect(DISTINCT i.text) AS instructions,
     collect(DISTINCT {req_id: nb.req_id, text: nb.text, rel_type: type(rel)}) AS neighbors
RETURN r.req_id AS req_id, r.text AS text, r.summary AS summary,
       r.type AS type, r.domain AS domain, r.communityId AS communityId,
       techniques, concepts, instructions, neighbors
"""

# Busca todos os vizinhos de 1 hop de um nó (qualquer tipo de relação semântica)
GET_GRAPH_NEIGHBORS = """
MATCH (r:Requirement {req_id: $req_id})
OPTIONAL MATCH (r)-[rel:RELATED_TO|DEPENDS_ON|EXTENDS|CONFLICTS_WITH|IMPLEMENTS|SIMILAR_TO]-(nb:Requirement)
WHERE ($graph_id = '' OR nb.graph_id = $graph_id)
RETURN nb.req_id AS req_id, nb.text AS text, nb.summary AS summary,
       nb.domain AS domain, type(rel) AS rel_type
LIMIT $limit
"""

GET_COMMUNITY_REQUIREMENTS = """
MATCH (r:Requirement {req_id: $req_id})
WHERE r.communityId IS NOT NULL
WITH r.communityId AS cid
MATCH (m:Requirement {communityId: cid})
RETURN m.req_id AS req_id, m.text AS text, m.summary AS summary,
       m.communityId AS community_id
ORDER BY m.req_id LIMIT $limit
"""

GET_COMMUNITY_REQUIREMENTS_FALLBACK = """
MATCH (r:Requirement {req_id: $req_id})-[:SIMILAR_TO]-(m:Requirement)
RETURN m.req_id AS req_id, m.text AS text, m.summary AS summary,
       m.communityId AS community_id
ORDER BY m.req_id LIMIT $limit
"""

GET_ALL_TECHNIQUES = """
MATCH (t:Technique)
RETURN t.tech_id AS id, t.name AS name, t.description AS description,
       t.category AS category ORDER BY t.tech_id
"""

GET_ALL_INSTRUCTIONS = """
MATCH (i:Instruction)
RETURN i.instr_id AS id, i.text AS text, i.context AS context ORDER BY i.instr_id
"""

GET_ALL_CONCEPTS = """
MATCH (c:Concept)
RETURN c.concept_id AS id, c.name AS name, c.definition AS definition ORDER BY c.concept_id
"""

GET_GRAPH_STATISTICS = """
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label
"""

COUNT_NODES = "MATCH (n) RETURN count(n) AS c"

CLEAR_DATABASE = "MATCH (n) DETACH DELETE n"

CREATE_REQUIREMENT = """
CREATE (r:Requirement {
    req_id: $req_id, text: $text, summary: $summary,
    type: $type, source: $source, domain: $domain,
    embedding: $embedding, embedding_model: $embedding_model,
    graph_id: $graph_id,
    embedding_ts: datetime(), created_at: datetime()
})
"""

CREATE_TECHNIQUE = """
CREATE (:Technique {
    tech_id: $tech_id, name: $name, description: $description,
    category: $category, source: $source,
    embedding: [], embedding_model: '', embedding_ts: datetime(), created_at: datetime()
})
"""

CREATE_INSTRUCTION = """
CREATE (:Instruction {
    instr_id: $instr_id, text: $text, context: $context, source: $source,
    embedding: [], embedding_model: '', embedding_ts: datetime(), created_at: datetime()
})
"""

CREATE_CONCEPT = """
CREATE (:Concept {
    concept_id: $concept_id, name: $name, definition: $definition, source: $source,
    embedding: [], embedding_model: '', embedding_ts: datetime(), created_at: datetime()
})
"""

CREATE_VECTOR_INDEX = """
CREATE VECTOR INDEX requirement_embeddings IF NOT EXISTS
FOR (r:Requirement) ON (r.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
}}
"""

GET_GRAPH_NODES_REQUIREMENTS = """
MATCH (r:Requirement)
WHERE $graph_id = '' OR r.graph_id = $graph_id
RETURN r.req_id AS id, r.text AS text, r.summary AS summary,
       r.type AS type, r.domain AS domain, r.communityId AS communityId
ORDER BY r.req_id LIMIT $limit
"""

GET_GRAPH_RELATIONSHIPS = """
MATCH (req:Requirement)-[r]->(b)
WHERE type(r) <> 'SIMILAR_TO'
  AND ($graph_id = '' OR req.graph_id = $graph_id)
RETURN req.req_id AS source,
       coalesce(b.req_id, b.tech_id, b.instr_id, b.concept_id) AS target,
       type(r) AS type
LIMIT 2000
"""

GET_SIMILAR_TO_SAMPLE = """
MATCH (a:Requirement)-[:SIMILAR_TO]->(b:Requirement)
WHERE ($graph_id = '' OR a.graph_id = $graph_id)
RETURN a.req_id AS source, b.req_id AS target LIMIT 300
"""

# --- Gestão de grafos múltiplos ---

LIST_GRAPHS = """
MATCH (g:Graph)
OPTIONAL MATCH (n) WHERE n.graph_id = g.graph_id AND NOT 'Graph' IN labels(n)
WITH g, count(n) AS node_count
RETURN g.graph_id AS graph_id, g.name AS name, node_count
ORDER BY g.created_at
"""

SET_MISSING_GRAPH_IDS = """
MATCH (n) WHERE n.graph_id IS NULL AND NOT 'Graph' IN labels(n)
SET n.graph_id = $graph_id
"""

CREATE_GRAPH_META = """
MERGE (g:Graph {graph_id: $graph_id})
ON CREATE SET g.created_at = datetime(), g.name = $name
ON MATCH SET g.name = $name, g.updated_at = datetime()
"""

COUNT_NODES_FOR_GRAPH = """
MATCH (n:Requirement {graph_id: $graph_id})
RETURN count(n) AS c
"""

SAMPLE_REQUIREMENTS_FOR_GRAPH = """
MATCH (r:Requirement)
WHERE (r.graph_id = 'default' OR r.graph_id IS NULL)
  AND ($keywords = [] OR any(kw IN $keywords WHERE toLower(r.text) CONTAINS kw
                              OR toLower(coalesce(r.domain,'')) CONTAINS kw))
WITH r ORDER BY rand()
LIMIT $limit
RETURN r.req_id AS req_id, r.text AS text, r.summary AS summary,
       r.type AS type, r.domain AS domain
"""

# Busca semântica via índice vetorial local (embedding_local, 384 dims)
# Requer que os nós tenham a propriedade embedding_local populada.
SEARCH_REQUIREMENTS_SEMANTIC = """
CALL db.index.vector.queryNodes('requirement_embeddings_local', $limit, $embedding)
YIELD node AS r, score
WHERE ($graph_id = '' OR r.graph_id = $graph_id)
RETURN r.req_id AS req_id, r.text AS text, r.summary AS summary,
       r.type AS type, r.domain AS domain, r.communityId AS communityId,
       score
"""

# Atualiza a propriedade embedding_local de um nó
SET_LOCAL_EMBEDDING = """
MATCH (r:Requirement {req_id: $req_id})
SET r.embedding_local = $embedding, r.embedding_local_model = $model
"""
