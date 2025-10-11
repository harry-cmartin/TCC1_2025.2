"""
Módulo para criar e popular o grafo de conhecimento de Engenharia de Requisitos no Neo4j
Baseado na especificação do arquivo Especificacao_do_grafo.md
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class Neo4jGraphCreator:
    """Classe responsável por criar e popular o grafo no Neo4j"""
    
    def __init__(self):
        self.uri = os.getenv('NEO4J_URL')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.database = os.getenv('NEO4J_DATABASE')
        self.driver = None
    
    def connect(self):
        """Conecta ao Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Testa a conexão
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print("✅ Conexão com Neo4j estabelecida com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro ao conectar com Neo4j: {e}")
            return False
    
    def close(self):
        """Fecha a conexão com Neo4j"""
        if self.driver:
            self.driver.close()
            print("🔌 Conexão com Neo4j fechada")
    
    def clear_database(self):
        """Limpa todos os nós e relacionamentos do banco"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Banco de dados limpo")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def create_requirement(self, req_id: str, text: str, summary: str = "", 
                          req_type: str = "funcional", source: str = "", 
                          domain: str = "", embedding: List[float] = None,
                          embedding_model: str = "text-embedding-3-small") -> str:
        """Cria um nó Requirement"""
        with self.driver.session(database=self.database) as session:
            query = """
            CREATE (r:Requirement {
                req_id: $req_id,
                text: $text,
                summary: $summary,
                type: $type,
                source: $source,
                domain: $domain,
                embedding: $embedding,
                embedding_model: $embedding_model,
                embedding_ts: datetime(),
                created_at: datetime()
            })
            RETURN r.req_id
            """
            result = session.run(query, {
                'req_id': req_id,
                'text': text,
                'summary': summary,
                'type': req_type,
                'source': source,
                'domain': domain,
                'embedding': embedding or [],
                'embedding_model': embedding_model
            })
            print(f"✅ Requirement criado: {req_id}")
            return result.single()[0]
    
    def create_technique(self, tech_id: str, name: str, description: str = "",
                        category: str = "", source: str = "", 
                        embedding: List[float] = None,
                        embedding_model: str = "text-embedding-3-small") -> str:
        """Cria um nó Technique"""
        with self.driver.session(database=self.database) as session:
            query = """
            CREATE (t:Technique {
                tech_id: $tech_id,
                name: $name,
                description: $description,
                category: $category,
                source: $source,
                embedding: $embedding,
                embedding_model: $embedding_model,
                embedding_ts: datetime(),
                created_at: datetime()
            })
            RETURN t.tech_id
            """
            result = session.run(query, {
                'tech_id': tech_id,
                'name': name,
                'description': description,
                'category': category,
                'source': source,
                'embedding': embedding or [],
                'embedding_model': embedding_model
            })
            print(f"✅ Technique criada: {name}")
            return result.single()[0]
    
    def create_instruction(self, instr_id: str, text: str, context: str = "",
                          source: str = "", embedding: List[float] = None,
                          embedding_model: str = "text-embedding-3-small") -> str:
        """Cria um nó Instruction"""
        with self.driver.session(database=self.database) as session:
            query = """
            CREATE (i:Instruction {
                instr_id: $instr_id,
                text: $text,
                context: $context,
                source: $source,
                embedding: $embedding,
                embedding_model: $embedding_model,
                embedding_ts: datetime(),
                created_at: datetime()
            })
            RETURN i.instr_id
            """
            result = session.run(query, {
                'instr_id': instr_id,
                'text': text,
                'context': context,
                'source': source,
                'embedding': embedding or [],
                'embedding_model': embedding_model
            })
            print(f"✅ Instruction criada: {instr_id}")
            return result.single()[0]
    
    def create_concept(self, concept_id: str, name: str, definition: str = "",
                      source: str = "", embedding: List[float] = None,
                      embedding_model: str = "text-embedding-3-small") -> str:
        """Cria um nó Concept"""
        with self.driver.session(database=self.database) as session:
            query = """
            CREATE (c:Concept {
                concept_id: $concept_id,
                name: $name,
                definition: $definition,
                source: $source,
                embedding: $embedding,
                embedding_model: $embedding_model,
                embedding_ts: datetime(),
                created_at: datetime()
            })
            RETURN c.concept_id
            """
            result = session.run(query, {
                'concept_id': concept_id,
                'name': name,
                'definition': definition,
                'source': source,
                'embedding': embedding or [],
                'embedding_model': embedding_model
            })
            print(f"✅ Concept criado: {name}")
            return result.single()[0]
    
    # Funções para criar relacionamentos
    def create_uses_technique_relationship(self, req_id: str, tech_id: str):
        """Cria relacionamento (:Requirement)-[:USES_TECHNIQUE]->(:Technique)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (t:Technique {tech_id: $tech_id})
            CREATE (r)-[:USES_TECHNIQUE]->(t)
            """
            session.run(query, {'req_id': req_id, 'tech_id': tech_id})
            print(f"✅ Relacionamento USES_TECHNIQUE criado: {req_id} -> {tech_id}")
    
    def create_refers_to_relationship(self, instr_id: str, concept_id: str):
        """Cria relacionamento (:Instruction)-[:REFERS_TO]->(:Concept)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (i:Instruction {instr_id: $instr_id})
            MATCH (c:Concept {concept_id: $concept_id})
            CREATE (i)-[:REFERS_TO]->(c)
            """
            session.run(query, {'instr_id': instr_id, 'concept_id': concept_id})
            print(f"✅ Relacionamento REFERS_TO criado: {instr_id} -> {concept_id}")
    
    def create_is_related_to_relationship(self, req_id: str, concept_id: str):
        """Cria relacionamento (:Requirement)-[:IS_RELATED_TO]->(:Concept)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (c:Concept {concept_id: $concept_id})
            CREATE (r)-[:IS_RELATED_TO]->(c)
            """
            session.run(query, {'req_id': req_id, 'concept_id': concept_id})
            print(f"✅ Relacionamento IS_RELATED_TO criado: {req_id} -> {concept_id}")
    
    def create_applies_to_relationship(self, tech_id: str, concept_id: str):
        """Cria relacionamento (:Technique)-[:APPLIES_TO]->(:Concept)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (t:Technique {tech_id: $tech_id})
            MATCH (c:Concept {concept_id: $concept_id})
            CREATE (t)-[:APPLIES_TO]->(c)
            """
            session.run(query, {'tech_id': tech_id, 'concept_id': concept_id})
            print(f"✅ Relacionamento APPLIES_TO criado: {tech_id} -> {concept_id}")
    
    def create_suggests_technique_relationship(self, instr_id: str, tech_id: str):
        """Cria relacionamento (:Instruction)-[:SUGGESTS_TECHNIQUE]->(:Technique)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (i:Instruction {instr_id: $instr_id})
            MATCH (t:Technique {tech_id: $tech_id})
            CREATE (i)-[:SUGGESTS_TECHNIQUE]->(t)
            """
            session.run(query, {'instr_id': instr_id, 'tech_id': tech_id})
            print(f"✅ Relacionamento SUGGESTS_TECHNIQUE criado: {instr_id} -> {tech_id}")
    
    def create_supported_by_relationship(self, req_id: str, instr_id: str):
        """Cria relacionamento (:Requirement)-[:SUPPORTED_BY]->(:Instruction)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (i:Instruction {instr_id: $instr_id})
            CREATE (r)-[:SUPPORTED_BY]->(i)
            """
            session.run(query, {'req_id': req_id, 'instr_id': instr_id})
            print(f"✅ Relacionamento SUPPORTED_BY criado: {req_id} -> {instr_id}")
    
    def populate_sample_data(self):
        """Popula o grafo com dados de exemplo baseados na especificação"""
        print("🎯 Iniciando criação de dados de exemplo...")
        
        # Criando Concepts
        self.create_concept("C001", "Requisito Funcional", 
                           "Um requisito que descreve uma função específica do sistema.",
                           "IEEE Std 830", [0.311, -0.244, 0.665])
        
        self.create_concept("C002", "Autenticação", 
                           "Processo de verificação da identidade de um usuário.",
                           "Literatura de Segurança", [0.521, -0.134, 0.445])
        
        self.create_concept("C003", "Stakeholder", 
                           "Qualquer pessoa ou organização que é afetada pelo sistema.",
                           "Sommerville (2011)", [0.221, -0.534, 0.775])
        
        # Criando Techniques
        self.create_technique("TECH001", "Entrevistas", 
                             "Coleta de requisitos por meio de entrevistas com stakeholders.",
                             "Elicitação", "Sommerville (2011)", [0.223, -0.316, 0.444])
        
        self.create_technique("TECH002", "Casos de Uso", 
                             "Técnica para capturar requisitos funcionais através de cenários.",
                             "Especificação", "Jacobson et al.", [0.423, -0.216, 0.644])
        
        self.create_technique("TECH003", "Prototipação", 
                             "Criação de versões preliminares do sistema para validação.",
                             "Validação", "Literatura de ES", [0.523, -0.116, 0.844])
        
        # Criando Instructions
        self.create_instruction("INST001", "Os requisitos devem ser claros e verificáveis.",
                               "Especificação", "Sommerville (2011)", [0.512, -0.122, 0.211])
        
        self.create_instruction("INST002", "Entrevistar stakeholders para elicitar requisitos.",
                               "Elicitação", "Kotonya & Sommerville", [0.612, -0.222, 0.311])
        
        self.create_instruction("INST003", "Validar requisitos com protótipos.",
                               "Validação", "Davis (1993)", [0.412, -0.322, 0.411])
        
        # Criando Requirements
        self.create_requirement("REQ001", 
                               "O sistema deve permitir que o usuário redefina a senha via e-mail.",
                               "Redefinição de senha por email", "funcional", "ProjetoX", "segurança",
                               [0.123, -0.456, 0.789])
        
        self.create_requirement("REQ002", 
                               "O sistema deve autenticar usuários através de login e senha.",
                               "Autenticação básica", "funcional", "ProjetoX", "segurança",
                               [0.223, -0.356, 0.889])
        
        self.create_requirement("REQ003", 
                               "O sistema deve responder em menos de 2 segundos.",
                               "Performance do sistema", "não-funcional", "ProjetoY", "performance",
                               [0.323, -0.256, 0.989])
        
        print("✅ Todos os nós criados!")
        
        # Criando relacionamentos
        print("🔗 Criando relacionamentos...")
        
        # Requirements relacionados a Concepts
        self.create_is_related_to_relationship("REQ001", "C002")  # Redefinir senha -> Autenticação
        self.create_is_related_to_relationship("REQ002", "C002")  # Login -> Autenticação
        self.create_is_related_to_relationship("REQ001", "C001")  # Req funcional -> Conceito funcional
        self.create_is_related_to_relationship("REQ002", "C001")  # Req funcional -> Conceito funcional
        
        # Instructions referem-se a Concepts
        self.create_refers_to_relationship("INST001", "C001")  # Requisitos verificáveis -> Req Funcional
        self.create_refers_to_relationship("INST002", "C003")  # Entrevistar -> Stakeholder
        
        # Techniques aplicam-se a Concepts
        self.create_applies_to_relationship("TECH001", "C003")  # Entrevistas -> Stakeholder
        self.create_applies_to_relationship("TECH002", "C001")  # Casos de uso -> Req Funcional
        self.create_applies_to_relationship("TECH003", "C001")  # Prototipação -> Req Funcional
        
        # Instructions sugerem Techniques
        self.create_suggests_technique_relationship("INST002", "TECH001")  # Entrevistar -> Entrevistas
        self.create_suggests_technique_relationship("INST003", "TECH003")  # Validar -> Prototipação
        
        # Requirements usam Techniques
        self.create_uses_technique_relationship("REQ001", "TECH001")  # Req senha -> Entrevistas
        self.create_uses_technique_relationship("REQ002", "TECH002")  # Req login -> Casos de uso
        
        # Requirements são suportados por Instructions
        self.create_supported_by_relationship("REQ001", "INST001")  # Req senha -> Verificável
        self.create_supported_by_relationship("REQ002", "INST001")  # Req login -> Verificável
        
        print("✅ Todos os relacionamentos criados!")
        print("🎉 Grafo de exemplo populado com sucesso!")
    
    def get_graph_statistics(self):
        """Retorna estatísticas básicas do grafo"""
        with self.driver.session(database=self.database) as session:
            # Conta nós por tipo
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as node_type, count(n) as count
                ORDER BY node_type
            """).data()
            
            # Conta relacionamentos por tipo
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY rel_type
            """).data()
            
            print("\n📊 Estatísticas do Grafo:")
            print("=" * 40)
            print("Nós por tipo:")
            for row in node_counts:
                print(f"  {row['node_type']}: {row['count']}")
            
            print("\nRelacionamentos por tipo:")
            for row in rel_counts:
                print(f"  {row['rel_type']}: {row['count']}")
            
            total_nodes = sum(row['count'] for row in node_counts)
            total_rels = sum(row['count'] for row in rel_counts)
            print(f"\nTotal: {total_nodes} nós, {total_rels} relacionamentos")


def main():
    """Função principal para executar a criação do grafo"""
    print("🚀 Iniciando criação do Grafo de Conhecimento de Engenharia de Requisitos")
    print("=" * 70)
    
    # Cria o grafo usando context manager
    with Neo4jGraphCreator() as graph:
        if not graph.driver:
            print("❌ Falha na conexão. Verifique se o Neo4j está rodando.")
            return
        
        # Limpa o banco (cuidado em produção!)
        print("\n🧹 Limpando banco de dados...")
        graph.clear_database()
        
        # Popula com dados de exemplo
        print("\n🎯 Criando dados de exemplo...")
        graph.populate_sample_data()
        
        # Mostra estatísticas
        graph.get_graph_statistics()
        
        print("\n✅ Grafo criado com sucesso!")
        print("🌐 Acesse o Neo4j Browser em: http://localhost:7474")
        print("🔍 Execute esta query para visualizar o grafo:")
        print("   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")


if __name__ == "__main__":
    main()