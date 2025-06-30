import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
from internal.entity_linker.entity_linker import EntityLinker
from typing import List
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TermExtractor:
    """
    Класс для извлечения терминов из текста с помощью NER-модели и последующей валидации.
    Использует модели из transformers и библиотеку natasha для морфологической обработки.
    """
    def __init__(self, model_path: str = os.getenv("MODEL_NER_PATH")):
        """
        Инициализация объекта TermExtractor.
        """
        if not model_path:
            logger.error("MODEL_PATH environment variable is not set")
            raise ValueError("MODEL_PATH is not set")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)

        self.linker = EntityLinker()

    def extract_terms(self, text: str) -> List[str]:
        """
        Извлекает термины из входного текста.

        Шаги обработки:
        1. Сегментация текста на предложения.
        2. Токенизация и предсказание меток NER для каждого предложения.
        3. Группировка меток B/I для выделения терминов.
        4. Очистка и фильтрация токенов.
        5. Валидация терминов лемматизацией.

        """
        text = text.encode().decode('utf-8', errors='ignore')
        
        doc = Doc(text)
        doc.segment(self.segmenter)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        all_terms = set()

        for sentence in sentences:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            word_ids = inputs.word_ids()
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs).logits

            predictions = torch.argmax(outputs, dim=2).squeeze(0)
            labels = [self.model.config.id2label[pred.item()] for pred in predictions]

            grouped_labels = []
            current_word = None
            current_group = []

            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != current_word:
                    if current_group:
                        grouped_labels.append(current_group[0])
                    current_group = [labels[i]]
                    current_word = word_id
                else:
                    current_group.append(labels[i])
            if current_group:
                grouped_labels.append(current_group[0])
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            word_tokens = []
            current_word = None
            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != current_word:
                    word_tokens.append(tokens[i])
                    current_word = word_id

            terms = []
            current_term = []
            for token, tag in zip(word_tokens, grouped_labels):
                if tag == 'B':
                    if current_term:
                        decoded_term = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(current_term))
                        terms.append(decoded_term)
                    current_term = [token]
                elif tag == 'I' and current_term:
                    current_term.append(token)
                else:
                    if current_term:
                        decoded_term = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(current_term))
                        terms.append(decoded_term)
                    current_term = []

            if current_term:
                decoded_term = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(current_term))
                terms.append(decoded_term)

            clean_terms = [
                t.replace('▁', '').replace('##', '').replace('Ġ', '').replace('-', ' ')
                .replace('(', '').replace(')', '').replace('.', '').strip()
                for t in terms if '�' not in t and t.strip()
            ]
            clean_terms = [
                t for t in clean_terms
                if len(t) > 3 and not t.startswith((' ', '(', ')', '.', '2', '5')) and any(c.isalpha() for c in t)
            ]

            valid_terms = self.validate_terms(text, clean_terms)
            all_terms.update(valid_terms)
        
        logger.debug(f"Extracted terms: {all_terms}")
        return list(all_terms)

    def lemmatize_text(self, text: str) -> str:
        """
        Лемматизирует входной текст.
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return " ".join(token.lemma for token in doc.tokens)

    def extract_sentences_with_terms(self, text: str, terms: list) -> dict:
        """
        Извлекает предложения из текста, содержащие заданные термины.
        """
        doc = Doc(text)
        doc.segment(self.segmenter)

        results = {}
        for term in terms:
            term_lemma = self.lemmatize_text(term)
            matching = []
            for sent in doc.sents:
                sent_lemma = self.lemmatize_text(sent.text)
                if term_lemma in sent_lemma:
                    matching.append(sent.text)
            results[term] = matching
        return results

    def validate_terms(self, text: str, terms: list) -> list:
        """
        Валидирует термины, проверяя наличие их лемматизированной формы в тексте.
        """
        text_lemma = self.lemmatize_text(text)
        valid_terms = []
        for term in terms:
            term_lemma = self.lemmatize_text(term)
            if term_lemma in text_lemma:
                valid_terms.append(term)
        return valid_terms
 