import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
from typing import List
import logging
from elasticsearch import Elasticsearch
import pymorphy3
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityLinker:
    """
    Класс для связывания терминов с сущностями в Elasticsearch и проверки их принадлежности к IT-сфере через SPARQL-запросы к Wikidata.
    """
    def __init__(self, 
                 es_url: str = os.getenv("ES_URL", "http://localhost:9200"), 
                 es_user: str = os.getenv("ES_USER", "elastic"), 
                 es_password: str = os.getenv("ES_PASSWORD", "SecurePass123"), 
                 sparql_url: str = os.getenv("SPARQL_URL", "https://query.wikidata.org/sparql"),
                 verify_certs: bool = False):
        """
        Инициализация клиента для Elasticsearch и SPARQLWrapper, а также лемматизатора pymorphy3.
        """
        if not all([es_url, es_user, es_password, sparql_url]):
            logger.error("Missing required environment variables (ES_URL, ES_USER, ES_PASSWORD, SPARQL_URL)")
            raise ValueError("One or more required environment variables are not set")

        self.morph = pymorphy3.MorphAnalyzer(lang='ru')
        self.it_entities = [
            "Q11660", "Q7397", "Q9143", "Q21198", "Q166142", "Q2539", "Q9135",
            "Q15869602", "Q5482740", "Q17103824", "Q627350", "Q937857", "Q12483",
            "Q230097", "Q1988917", "Q1077784", "Q11661", "Q4897819", "Q185571",
            "Q431289", "Q8072", "Q220796", "Q1520851", "Q9089", "Q6563", "Q138793",
            "Q208620", "Q1751403", "Q2053524", "Q2848294", "Q209788", "Q1944715"
        ]
    
        try:
            self.es = Elasticsearch(
                es_url,
                basic_auth=(es_user, es_password),
                verify_certs=verify_certs
            )
            logger.info(f"Elasticsearch client initialized with URL: {es_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}")
            raise

        self.sparql_url = sparql_url

    def lemmatize(self, text: str) -> str:
        """
        Лемматизирует входной текст, разделяя по пробелам и приводя каждое слово к нормальной форме.
        """
        words = text.split()
        lemmas = [self.morph.parse(word)[0].normal_form for word in words]
        return " ".join(lemmas)

    def search_elasticsearch(self, term: str) -> list:
        """
        Поиск терминов в индексе Elasticsearch с использованием лемматизации и трех типов запросов:
        exact term, phrase match и fuzzy match.
        """
        lemmatized_query = self.lemmatize(term)
        try:
            response = self.es.search(
                index="terms",
                query={
                    "bool": {
                        "should": [
                            {"term": {"label.value.keyword": {"value": lemmatized_query, "boost": 5}}},
                            {"match_phrase": {"label.value": {"query": lemmatized_query, "slop": 1, "boost": 3}}},
                            {"match": {"label.value": {"query": lemmatized_query, "fuzziness": "AUTO", "prefix_length": 2, "minimum_should_match": "75%"}}}
                        ]
                    }
                },
                size=3
            )
            hits = response["hits"]["hits"]

            if len(hits) > 1 and hits[0]["_score"] > hits[1]["_score"] * 2:
                hits = hits[:1]
            logger.debug(f"Elasticsearch search for term '{term}' returned {len(hits)} hits")
            return hits
        except Exception as e:
            logger.error(f"Error searching Elasticsearch for term '{term}': {e}")
            return []

    def check_it_related(self, entities: list) -> list:
        """
        Проверяет, относятся ли переданные сущности к IT-сфере,
        выполняя SPARQL-запрос к Wikidata по свойствам P31, P279
        """
        if not entities:
            logger.warning("No entities provided for IT-related check")
            return []
        values_entities = " ".join(f"wd:{entity}" for entity in entities)
        values_targets = " ".join(f"wd:{t}" for t in self.it_entities)
        
        query = f"""
        SELECT ?entity WHERE {{
          VALUES ?entity {{ {values_entities} }}
          VALUES ?target {{ {values_targets} }}
          ?entity (wdt:P31|wdt:P279)+ ?target .
        }}
        """
        
        try:
            sparql = SPARQLWrapper(self.sparql_url)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            it_related_uris = [binding['entity']['value'] for binding in results['results']['bindings']]
            it_related_ids = [uri.split('/')[-1] for uri in it_related_uris]
            logger.debug(f"SPARQL query returned {len(it_related_ids)} IT-related entities")
            return it_related_ids
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return []

    def link_entities(self, term: str) -> list:
        """
        Связывает термин с сущностями: выполняет поиск в Elasticsearch,
        фильтрует по IT-тематике и возвращает связанные сущности.
        """
        hits = self.search_elasticsearch(term)
        if not hits:
            logger.warning(f"No documents found in Elasticsearch for term '{term}'")
            return []
        
        entity_ids = [hit["_source"].get("id") for hit in hits if "id" in hit["_source"]]
        if not entity_ids:
            logger.warning(f"No entity IDs extracted for term '{term}'")
            return []
        it_related_ids = self.check_it_related(entity_ids)
        linked_entities = [hit for hit in hits if hit["_source"].get("id") in it_related_ids]
        logger.info(f"Linked {len(linked_entities)} entities for term '{term}'")
        
        return linked_entities
