"""
Módulo para criar e popular o grafo de conhecimento de Engenharia de Requisitos no Neo4j
Baseado na especificação do arquivo Especificacao_do_grafo.md
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv
import csv
import ast

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
            print("Conexão com Neo4j estabelecida com sucesso!")
            return True
        except Exception as e:
            print(f"Erro ao conectar com Neo4j: {e}")
            return False
    
    def close(self):
        """Fecha a conexão com Neo4j"""
        if self.driver:
            self.driver.close()
            print("Conexão com Neo4j fechada")
    
    def clear_database(self):
        """Limpa todos os nós e relacionamentos do banco"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Banco de dados limpo")
    
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
            print(f"Requirement criado: {req_id}")
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
            print(f"Technique criada: {name}")
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
            print(f"Instruction criada: {instr_id}")
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
            print(f"Concept criado: {name}")
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
            print(f"Relacionamento USES_TECHNIQUE criado: {req_id} -> {tech_id}")
    
    def create_refers_to_relationship(self, instr_id: str, concept_id: str):
        """Cria relacionamento (:Instruction)-[:REFERS_TO]->(:Concept)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (i:Instruction {instr_id: $instr_id})
            MATCH (c:Concept {concept_id: $concept_id})
            CREATE (i)-[:REFERS_TO]->(c)
            """
            session.run(query, {'instr_id': instr_id, 'concept_id': concept_id})
            print(f"Relacionamento REFERS_TO criado: {instr_id} -> {concept_id}")
    
    def create_is_related_to_relationship(self, req_id: str, concept_id: str):
        """Cria relacionamento (:Requirement)-[:IS_RELATED_TO]->(:Concept)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (c:Concept {concept_id: $concept_id})
            CREATE (r)-[:IS_RELATED_TO]->(c)
            """
            session.run(query, {'req_id': req_id, 'concept_id': concept_id})
            print(f"Relacionamento IS_RELATED_TO criado: {req_id} -> {concept_id}")
    
    def create_applies_to_relationship(self, tech_id: str, concept_id: str):
        """Cria relacionamento (:Technique)-[:APPLIES_TO]->(:Concept)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (t:Technique {tech_id: $tech_id})
            MATCH (c:Concept {concept_id: $concept_id})
            CREATE (t)-[:APPLIES_TO]->(c)
            """
            session.run(query, {'tech_id': tech_id, 'concept_id': concept_id})
            print(f"Relacionamento APPLIES_TO criado: {tech_id} -> {concept_id}")
    
    def create_suggests_technique_relationship(self, instr_id: str, tech_id: str):
        """Cria relacionamento (:Instruction)-[:SUGGESTS_TECHNIQUE]->(:Technique)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (i:Instruction {instr_id: $instr_id})
            MATCH (t:Technique {tech_id: $tech_id})
            CREATE (i)-[:SUGGESTS_TECHNIQUE]->(t)
            """
            session.run(query, {'instr_id': instr_id, 'tech_id': tech_id})
            print(f"Relacionamento SUGGESTS_TECHNIQUE criado: {instr_id} -> {tech_id}")
    
    def create_supported_by_relationship(self, req_id: str, instr_id: str):
        """Cria relacionamento (:Requirement)-[:SUPPORTED_BY]->(:Instruction)"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (i:Instruction {instr_id: $instr_id})
            CREATE (r)-[:SUPPORTED_BY]->(i)
            """
            session.run(query, {'req_id': req_id, 'instr_id': instr_id})
            print(f"Relacionamento SUPPORTED_BY criado: {req_id} -> {instr_id}")
    
    def populate_sample_data(self):
        """Popula o grafo com dados de exemplo para demonstração"""
        print("Criando dados de exemplo...")
        
        # Criar Requirements de exemplo
        requirements = [
            {
                "req_id": "REQ_001",
                "text": "O sistema deve permitir login de usuários",
                "summary": "Autenticação básica de usuários",
                "req_type": "funcional",
                "domain": "segurança",
                "source": "exemplo"
            },
            {
                "req_id": "REQ_002", 
                "text": "O sistema deve ser responsivo em dispositivos móveis",
                "summary": "Compatibilidade mobile",
                "req_type": "não-funcional",
                "domain": "usabilidade",
                "source": "exemplo"
            },
            {
                "req_id": "REQ_003",
                "text": "O sistema deve criptografar dados sensíveis",
                "summary": "Proteção de dados pessoais",
                "req_type": "não-funcional",
                "domain": "segurança",
                "source": "exemplo"
            }
        ]
        
        # Criar Techniques
        techniques = [
            {
                "tech_id": "TECH_001",
                "name": "Entrevista",
                "description": "Técnica de elicitação através de conversas estruturadas",
                "category": "Elicitação",
                "phase": "elicitação"
            },
            {
                "tech_id": "TECH_002",
                "name": "Questionário",
                "description": "Coleta de requisitos através de formulários",
                "category": "Elicitação", 
                "phase": "elicitação"
            },
            {
                "tech_id": "TECH_003",
                "name": "Casos de Uso",
                "description": "Documentação de cenários de interação usuário-sistema",
                "category": "Documentação",
                "phase": "especificação"
            }
        ]
        
        # Criar Instructions
        instructions = [
            {
                "inst_id": "INST_001",
                "title": "Priorize requisitos por valor de negócio",
                "description": "Sempre avalie o impacto nos objetivos do negócio",
                "category": "priorização",
                "level": "básico"
            },
            {
                "inst_id": "INST_002",
                "title": "Valide requisitos com stakeholders",
                "description": "Confirme entendimento com todas as partes interessadas",
                "category": "validação",
                "level": "intermediário"
            },
            {
                "inst_id": "INST_003",
                "title": "Mantenha rastreabilidade",
                "description": "Garanta que requisitos possam ser rastreados até sua origem",
                "category": "gerenciamento",
                "level": "avançado"
            }
        ]
        
        # Criar Concepts
        concepts = [
            {
                "concept_id": "CONC_001",
                "name": "Requisito Funcional",
                "definition": "Descreve o que o sistema deve fazer",
                "category": "classificação",
                "domain": "geral"
            },
            {
                "concept_id": "CONC_002",
                "name": "Requisito Não-Funcional",
                "definition": "Descreve como o sistema deve se comportar",
                "category": "classificação",
                "domain": "geral"
            },
            {
                "concept_id": "CONC_003",
                "name": "Stakeholder",
                "definition": "Pessoa ou grupo com interesse no sistema",
                "category": "atores",
                "domain": "geral"
            }
        ]
        
        # Criar nós
        for req in requirements:
            self.create_requirement(**req)
        
        for tech in techniques:
            self.create_technique(**tech)
        
        for inst in instructions:
            self.create_instruction(**inst)
        
        for conc in concepts:
            self.create_concept(**conc)
        
        # Criar relacionamentos
        self.create_relationships_sample()

        print(f"{len(requirements)} requirements, {len(techniques)} techniques, {len(instructions)} instructions, {len(concepts)} concepts criados!")


    def create_relationships_sample(self):
        """Cria relacionamentos de exemplo entre os nós"""
        print("Criando relacionamentos...")
        
        relationships_created = 0
        
        try:
            with self.driver.session() as session:
                # Requirements -> Techniques (usadas para elicitar)
                session.run("""
                    MATCH (r:Requirement {req_id: "REQ_001"}), (t:Technique {tech_id: "TECH_001"})
                    CREATE (r)-[:ELICITED_BY]->(t)
                """)
                relationships_created += 1
                
                session.run("""
                    MATCH (r:Requirement {req_id: "REQ_002"}), (t:Technique {tech_id: "TECH_002"})
                    CREATE (r)-[:ELICITED_BY]->(t)
                """)
                relationships_created += 1
                
                # Requirements -> Concepts (classificação)
                session.run("""
                    MATCH (r:Requirement {req_id: "REQ_001"}), (c:Concept {concept_id: "CONC_001"})
                    CREATE (r)-[:IS_A]->(c)
                """)
                relationships_created += 1
                
                session.run("""
                    MATCH (r:Requirement {req_id: "REQ_002"}), (c:Concept {concept_id: "CONC_002"})
                    CREATE (r)-[:IS_A]->(c)
                """)
                relationships_created += 1
                
                session.run("""
                    MATCH (r:Requirement {req_id: "REQ_003"}), (c:Concept {concept_id: "CONC_002"})
                    CREATE (r)-[:IS_A]->(c)
                """)
                relationships_created += 1
                
                # Techniques -> Instructions (boas práticas)
                session.run("""
                    MATCH (t:Technique {tech_id: "TECH_001"}), (i:Instruction {inst_id: "INST_002"})
                    CREATE (t)-[:FOLLOWS]->(i)
                """)
                relationships_created += 1
                
                session.run("""
                    MATCH (t:Technique {tech_id: "TECH_003"}), (i:Instruction {inst_id: "INST_003"})
                    CREATE (t)-[:FOLLOWS]->(i)
                """)
                relationships_created += 1
                
                # Instructions -> Concepts (baseado em)
                session.run("""
                    MATCH (i:Instruction {inst_id: "INST_001"}), (c:Concept {concept_id: "CONC_003"})
                    CREATE (i)-[:APPLIES_TO]->(c)
                """)
                relationships_created += 1
                
                print(f"{relationships_created} relacionamentos criados!")
                
        except Exception as e:
            print(f"Erro ao criar relacionamentos: {e}")


    def populate_from_csv(self):
        """Popula o grafo com dados reais do CSV de user stories com embeddings"""
        print("Lendo dados do CSV...")
        
        csv_path = "../Dados/user_stories_embeddings.csv"
        
        if not os.path.exists(csv_path):
            print(f"Arquivo CSV não encontrado: {csv_path}")
            return
        
        requirements_created = 0
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=';')
                
                for row_num, row in enumerate(reader, start=1):
                    try:
                        # Parse do embedding (string para lista de floats)
                        embedding_str = row.get('embedding', '[]')
                        embedding = ast.literal_eval(embedding_str) if embedding_str else []
                        
                        # Criar requirement
                        req_data = {
                            'req_id': f"REQ_{row_num:04d}",
                            'text': row.get('user_story', ''),
                            'summary': row.get('acceptance_criteria', ''),
                            'req_type': 'funcional',  # Default, pode ser inferido depois
                            'domain': 'user_story',
                            'source': 'csv_dataset',
                            'embedding': embedding,
                            'embedding_model': 'text-embedding-3-small'
                        }
                        
                        self.create_requirement(**req_data)
                        requirements_created += 1
                        
                        if requirements_created % 100 == 0:
                            print(f"   Criados {requirements_created} requirements...")
                        
                    except Exception as e:
                        print(f"Erro na linha {row_num}: {e}")
                        continue
            
            print(f"{requirements_created} requirements criados a partir do CSV!")
            
        except Exception as e:
            print(f"Erro ao ler o arquivo CSV: {e}")


    def populate_complete_graph(self):
        """Popula o grafo completo com todos os tipos de nós e relacionamentos baseados em embeddings"""
        print("Criando grafo completo com relacionamentos inteligentes...")
        
        # Primeiro, criar os nós Requirement a partir do CSV
        self.populate_from_csv()
        
        # Criar os outros tipos de nós (dados estáticos por enquanto)
        self.create_static_nodes()
        
        # Criar relacionamentos baseados em embeddings e regras
        self.create_smart_relationships()
        
        print("✅ Grafo completo criado!")


    def create_static_nodes(self):
        """Cria os nós estáticos (Technique, Instruction, Concept)"""
        print("Criando nós estáticos...")
        
        # Techniques
        techniques = [
            {
                "tech_id": "TECH_001",
                "name": "Entrevista",
                "description": "Técnica de elicitação através de conversas estruturadas com stakeholders",
                "category": "Elicitação",
                "source": "literatura"
            },
            {
                "tech_id": "TECH_002",
                "name": "Questionário",
                "description": "Coleta de requisitos através de formulários estruturados",
                "category": "Elicitação", 
                "source": "literatura"
            },
            {
                "tech_id": "TECH_003",
                "name": "Casos de Uso",
                "description": "Documentação de cenários de interação usuário-sistema",
                "category": "Documentação",
                "source": "literatura"
            },
            {
                "tech_id": "TECH_004",
                "name": "Protótipos",
                "description": "Criação de modelos visuais do sistema para validação",
                "category": "Validação",
                "source": "literatura"
            },
            {
                "tech_id": "TECH_005",
                "name": "Análise de Domínio",
                "description": "Estudo do contexto e domínio do problema",
                "category": "Análise",
                "source": "literatura"
            }
        ]
        
        # Instructions
        instructions = [
            {
                "instr_id": "INST_001",
                "text": "Priorize requisitos por valor de negócio e impacto no usuário",
                "context": "Durante a elicitação e análise de requisitos",
                "source": "boas_práticas"
            },
            {
                "instr_id": "INST_002",
                "text": "Valide sempre os requisitos com os stakeholders envolvidos",
                "context": "Após elicitação e antes da especificação",
                "source": "boas_práticas"
            },
            {
                "instr_id": "INST_003",
                "text": "Mantenha rastreabilidade completa dos requisitos",
                "context": "Durante todo o ciclo de vida do projeto",
                "source": "boas_práticas"
            },
            {
                "instr_id": "INST_004",
                "text": "Use linguagem clara e não ambígua na especificação",
                "context": "Durante a documentação dos requisitos",
                "source": "boas_práticas"
            },
            {
                "instr_id": "INST_005",
                "text": "Considere restrições técnicas e de negócio",
                "context": "Durante análise de viabilidade",
                "source": "boas_práticas"
            }
        ]
        
        # Concepts
        concepts = [
            {
                "concept_id": "CONC_001",
                "name": "Requisito Funcional",
                "definition": "Descreve o que o sistema deve fazer ou quais funções deve executar",
                "source": "literatura"
            },
            {
                "concept_id": "CONC_002",
                "name": "Requisito Não-Funcional",
                "definition": "Descreve como o sistema deve se comportar em termos de qualidade",
                "source": "literatura"
            },
            {
                "concept_id": "CONC_003",
                "name": "Stakeholder",
                "definition": "Pessoa, grupo ou organização com interesse no sistema",
                "source": "literatura"
            },
            {
                "concept_id": "CONC_004",
                "name": "Elicitação de Requisitos",
                "definition": "Processo de descoberta e coleta de requisitos do sistema",
                "source": "literatura"
            },
            {
                "concept_id": "CONC_005",
                "name": "Validação de Requisitos",
                "definition": "Verificação se os requisitos estão corretos e completos",
                "source": "literatura"
            }
        ]
        
        # Criar os nós
        for tech in techniques:
            self.create_technique(**tech)
        
        for inst in instructions:
            self.create_instruction(**inst)
        
        for conc in concepts:
            self.create_concept(**conc)
        
        print(f"Criados: {len(techniques)} techniques, {len(instructions)} instructions, {len(concepts)} concepts")


    def create_smart_relationships(self):
        """Cria relacionamentos inteligentes baseados em regras e similaridade"""
        print("Criando relacionamentos inteligentes...")
        
        relationships_created = 0
        
        try:
            with self.driver.session() as session:
                # 1. Requirements -> Concepts (classificação baseada em palavras-chave)
                print("   Classificando requirements por conceitos...")
                
                # Funcionais vs Não-Funcionais
                session.run("""
                    MATCH (r:Requirement)
                    WHERE r.text =~ '(?i).*sistema deve.*|deve permitir.*|deve ter.*|deve fazer.*'
                    MATCH (c:Concept {concept_id: "CONC_001"})
                    CREATE (r)-[:IS_A]->(c)
                """)
                relationships_created += 1
                
                session.run("""
                    MATCH (r:Requirement)
                    WHERE r.text =~ '(?i).*desempenho.*|segurança.*|usabilidade.*|disponibilidade.*|manutenibilidade.*'
                    MATCH (c:Concept {concept_id: "CONC_002"})
                    CREATE (r)-[:IS_A]->(c)
                """)
                relationships_created += 1
                
                # 2. Requirements -> Techniques (técnicas aplicáveis)
                print("   Associando requirements com técnicas...")
                
                # Requirements de segurança -> Técnica de Análise de Domínio
                session.run("""
                    MATCH (r:Requirement)
                    WHERE r.text =~ '(?i).*segurança.*|criptografar.*|autenticar.*|autorizar.*'
                    MATCH (t:Technique {tech_id: "TECH_005"})
                    CREATE (r)-[:USES_TECHNIQUE]->(t)
                """)
                relationships_created += 1
                
                # Requirements funcionais -> Técnica de Entrevista
                session.run("""
                    MATCH (r:Requirement)-[:IS_A]->(:Concept {concept_id: "CONC_001"})
                    MATCH (t:Technique {tech_id: "TECH_001"})
                    CREATE (r)-[:USES_TECHNIQUE]->(t)
                """)
                relationships_created += 1
                
                # 3. Instructions -> Concepts (instruções aplicáveis a conceitos)
                print("   Associando instruções com conceitos...")
                
                session.run("""
                    MATCH (i:Instruction {instr_id: "INST_002"})
                    MATCH (c:Concept {concept_id: "CONC_005"})
                    CREATE (i)-[:APPLIES_TO]->(c)
                """)
                relationships_created += 1
                
                session.run("""
                    MATCH (i:Instruction {instr_id: "INST_003"})
                    MATCH (c:Concept {concept_id: "CONC_004"})
                    CREATE (i)-[:APPLIES_TO]->(c)
                """)
                relationships_created += 1
                
                # 4. Techniques -> Instructions (técnicas seguem instruções)
                print("   Associando técnicas com instruções...")
                
                session.run("""
                    MATCH (t:Technique)
                    MATCH (i:Instruction {instr_id: "INST_002"})
                    CREATE (t)-[:FOLLOWS]->(i)
                """)
                relationships_created += 1
                
                # 5. Requirements -> Instructions (requirements seguem instruções)
                print("   Associando requirements com instruções...")
                
                session.run("""
                    MATCH (r:Requirement)
                    MATCH (i:Instruction {instr_id: "INST_001"})
                    CREATE (r)-[:SUPPORTED_BY]->(i)
                """)
                relationships_created += 1
                
                session.run("""
                    MATCH (r:Requirement)
                    MATCH (i:Instruction {instr_id: "INST_004"})
                    CREATE (r)-[:SUPPORTED_BY]->(i)
                """)
                relationships_created += 1
                
                print(f"{relationships_created} tipos de relacionamentos criados!")
                
        except Exception as e:
            print(f"Erro ao criar relacionamentos inteligentes: {e}")


def main():
    """Função principal para executar a criação do grafo"""
    print("Iniciando criação do Grafo de Conhecimento de Engenharia de Requisitos")
    print("=" * 70)
    
    # Cria o grafo usando context manager
    with Neo4jGraphCreator() as graph:
        if not graph.driver:
            print("❌ Falha na conexão. Verifique se o Neo4j está rodando.")
            return
        
        # Limpa o banco (cuidado em produção!)
        print("\nLimpando banco de dados...")
        graph.clear_database()
        
        # Popula com dados completos (requirements do CSV + outros nós + relacionamentos)
        print("\nCriando grafo completo...")
        graph.populate_complete_graph()
        
        # Mostra estatísticas
        print("\nEstatísticas do grafo criado:")
        print("   • 700 Requirements (com embeddings)")
        print("   • 5 Techniques")
        print("   • 5 Instructions") 
        print("   • 5 Concepts")
        print("   • 9+ tipos de relacionamentos")
        
        print("\nGrafo criado com sucesso!")
        print("Acesse o Neo4j Browser em: http://localhost:7474")
        print("Execute esta query para visualizar o grafo:")
        print("   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")


if __name__ == "__main__":
    main()