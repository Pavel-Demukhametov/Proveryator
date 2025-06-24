from elasticsearch import Elasticsearch
import pymorphy3
import os
from SPARQLWrapper import SPARQLWrapper, JSON

class EntityLinker:
    def __init__(self, es_url: str = "http://localhost:9200", es_user: str = "elastic", 
                 es_password: str = "SecurePass123", verify_certs: bool = False):

        self.morph = pymorphy3.MorphAnalyzer(lang='ru')
        self.it_entities = [
            "Q11660", "Q7397", "Q9143", "Q21198", "Q166142", "Q2539", "Q9135",
            "Q15869602", "Q5482740", "Q17103824", "Q627350", "Q937857", "Q12483",
            "Q230097", "Q1988917", "Q1077784", "Q11661", "Q4897819", "Q185571",
            "Q431289", "Q8072", "Q220796", "Q1520851", "Q9089", "Q6563", "Q138793",
            "Q208620", "Q1751403", "Q2053524", "Q2848294", "Q209788", "Q1944715"
        ]
    
        self.es = Elasticsearch(
            os.getenv("ES_URL", es_url),
            basic_auth=(os.getenv("ES_USER", es_user), es_password),
            verify_certs=verify_certs
        )

    def lemmatize(self, text: str) -> str:
        words = text.split()
        lemmas = [self.morph.parse(word)[0].normal_form for word in words]
        return " ".join(lemmas)

    def search_elasticsearch(self, term: str) -> list:
        lemmatized_query = self.lemmatize(term)
        response = self.es.search(
            index="terms",
            query={
                "bool": {
                    "should": [
                        {
                            "term": {
                                "label.value.keyword": {
                                    "value": lemmatized_query,
                                    "boost": 5
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "label.value": {
                                    "query": lemmatized_query,
                                    "slop": 1,
                                    "boost": 3
                                }
                            }
                        },
                        {
                            "match": {
                                "label.value": {
                                    "query": lemmatized_query,
                                    "fuzziness": "AUTO",
                                    "prefix_length": 2,
                                    "minimum_should_match": "75%"
                                }
                            }
                        }
                    ]
                }
            },
            size=3
        )
        
        hits = response["hits"]["hits"]
        if len(hits) > 1 and hits[0]["_score"] > hits[1]["_score"] * 2:
            hits = hits[:1]
        
        return hits

    def check_it_related(self, entities: list) -> list:
        if not entities:
            return []
        
        values_entities = " ".join(f"wd:{entity}" for entity in entities)
        values_targets = " ".join(f"wd:{t}" for t in self.it_entities)
        
        query = f"""
        SELECT ?entity WHERE {{
          VALUES ?entity {{ {values_entities} }}
          VALUES ?target {{ {values_targets} }}
          ?entity (wdt:P31|wdt:P279|wdt:P361)+ ?target .
        }}
        """
        
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        try:
            results = sparql.query().convert()
            it_related_uris = [binding['entity']['value'] for binding in results['results']['bindings']]
            it_related_ids = [uri.split('/')[-1] for uri in it_related_uris]
            return it_related_ids
        except Exception as e:
            print(f"Ошибка SPARQL-запроса: {e}")
            return []

    def link_entities(self, term: str) -> list:
        hits = self.search_elasticsearch(term)
        if not hits:
            print("Не найдено документов в Elasticsearch.")
            return []
        
        entity_ids = [hit["_source"].get("id") for hit in hits if "id" in hit["_source"]]
        if not entity_ids:
            print("Не удалось извлечь ID сущностей.")
            return []
        
        it_related_ids = self.check_it_related(entity_ids)
        linked_entities = [hit for hit in hits if hit["_source"].get("id") in it_related_ids]
        
        return linked_entities
