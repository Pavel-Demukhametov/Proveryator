from gensim.models import KeyedVectors
from natasha import Doc, Segmenter, NewsEmbedding
import nltk
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
import numpy as np
from typing import List, Optional

nltk.download('stopwords')

class EmbeddingManager:
    """
    Класс для создания эмбеддингов и проверки похожести.
    """
    def __init__(self):
        self.segmenter = Segmenter()
        self.stop_words = set(stopwords.words('russian'))
        self.punct = {',', '?', '!', '.', ';', ':', '...', '(', ')', '\'', '\"', '«', '»'}
        self.emb_model = self._load_embedding_model()

    def _load_embedding_model(self) -> KeyedVectors:
        news_emb = NewsEmbedding()
        return self._as_gensim(news_emb)

    @staticmethod
    def _as_gensim(navec_model) -> KeyedVectors:
        model = KeyedVectors(vector_size=navec_model.pq.dim)
        weights = navec_model.pq.unpack()
        model.add_vectors(navec_model.vocab.words, weights)
        return model

    def tokenize(self, text: str) -> List[str]:
        doc = Doc(text)
        doc.segment(self.segmenter)
        return [token.text.lower() for token in doc.tokens]

    def model_words_filter(self, tokens: List[str]) -> List[str]:
        return [
            token for token in tokens
            if token in self.emb_model and token not in self.stop_words and token not in self.punct
        ]

    def get_sentence_embedding(self, text: str) -> Optional[np.ndarray]:
        tokens = self.tokenize(text)
        filtered_tokens = self.model_words_filter(tokens)
        if not filtered_tokens:
            return None
        vectors = [self.emb_model[token] for token in filtered_tokens if token in self.emb_model]
        if not vectors:
            return None
        embedding = np.mean(vectors, axis=0)
        return embedding

    def is_similar(
        self,
        existing_embeddings: List[np.ndarray],
        new_embedding: np.ndarray,
        threshold: float = 0.8
    ) -> bool:
        for emb in existing_embeddings:
            similarity = 1 - cosine(emb, new_embedding)
            if similarity >= threshold:
                return True
        return False