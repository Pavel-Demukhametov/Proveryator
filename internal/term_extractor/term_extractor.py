# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
# from internal.entity_linker.entity_linker import EntityLinker
# from typing import List
# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
# from internal.entity_linker.entity_linker import EntityLinker


# class TermExtractor:
#     def __init__(self, model_path='D:\\Program Files\\Lecture_test_front\\fast\\ruSciBert-NER'):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
#         self.model = AutoModelForTokenClassification.from_pretrained(model_path)

#         self.segmenter = Segmenter()
#         self.emb = NewsEmbedding()
#         self.morph_vocab = MorphVocab()
#         self.morph_tagger = NewsMorphTagger(self.emb)
#         self.linker = EntityLinker()

#     def extract_terms(self, text: str) -> List[str]:
#         text = text.encode().decode('utf-8', errors='ignore')
        
#         doc = Doc(text)
#         doc.segment(self.segmenter)
#         sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

#         all_terms = set()

#         for sentence in sentences:
#             inputs = self.tokenizer(
#                 sentence,
#                 return_tensors="pt",
#                 truncation=True,
#                 padding=True,
#                 max_length=512
#             )
#             word_ids = inputs.word_ids()
#             inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#             with torch.no_grad():
#                 outputs = self.model(**inputs).logits

#             predictions = torch.argmax(outputs, dim=2).squeeze(0)
#             labels = [self.model.config.id2label[pred.item()] for pred in predictions]

#             grouped_labels = []
#             current_word = None
#             current_group = []

#             for i, word_id in enumerate(word_ids):
#                 if word_id is None:
#                     continue
#                 if word_id != current_word:
#                     if current_group:
#                         grouped_labels.append(current_group[0])
#                     current_group = [labels[i]]
#                     current_word = word_id
#                 else:
#                     current_group.append(labels[i])
#             if current_group:
#                 grouped_labels.append(current_group[0])

#             tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
#             word_tokens = []
#             current_word = None
#             for i, word_id in enumerate(word_ids):
#                 if word_id is None:
#                     continue
#                 if word_id != current_word:
#                     word_tokens.append(tokens[i])
#                     current_word = word_id

#             terms = []
#             current_term = []
#             for token, tag in zip(word_tokens, grouped_labels):
#                 if tag == 'B':
#                     if current_term:
#                         decoded_term = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(current_term))
#                         terms.append(decoded_term)
#                     current_term = [token]
#                 elif tag == 'I' and current_term:
#                     current_term.append(token)
#                 else:
#                     if current_term:
#                         decoded_term = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(current_term))
#                         terms.append(decoded_term)
#                     current_term = []

#             if current_term:
#                 decoded_term = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(current_term))
#                 terms.append(decoded_term)

#             clean_terms = [
#                 t.replace('▁', '').replace('##', '').replace('Ġ', '').replace('-', ' ')
#                 .replace('(', '').replace(')', '').replace('.', '').strip()
#                 for t in terms if '�' not in t and t.strip()
#             ]
#             clean_terms = [
#                 t for t in clean_terms
#                 if len(t) > 3 and not t.startswith((' ', '(', ')', '.', '2', '5')) and any(c.isalpha() for c in t)
#             ]

#             valid_terms = self.validate_terms(text, clean_terms)
#             all_terms.update(valid_terms)
#         print(all_terms)
#         return list(all_terms)
#     def lemmatize_text(self, text: str) -> str:
#         doc = Doc(text)
#         doc.segment(self.segmenter)
#         doc.tag_morph(self.morph_tagger)
#         for token in doc.tokens:
#             token.lemmatize(self.morph_vocab)
#         return " ".join(token.lemma for token in doc.tokens)

#     def extract_sentences_with_terms(self, text: str, terms: list) -> dict:
#         doc = Doc(text)
#         doc.segment(self.segmenter)

#         results = {}
#         for term in terms:
#             term_lemma = self.lemmatize_text(term)
#             matching = []
#             for sent in doc.sents:
#                 sent_lemma = self.lemmatize_text(sent.text)
#                 if term_lemma in sent_lemma:
#                     matching.append(sent.text)
#             results[term] = matching
#         return results

#     def validate_terms(self, text: str, terms: list) -> list:
#         text_lemma = self.lemmatize_text(text)
#         valid_terms = []
#         for term in terms:
#             term_lemma = self.lemmatize_text(term)
#             if term_lemma in text_lemma:
#                 valid_terms.append(term)
#         return valid_terms
from typing import List
from g4f.client import Client
from g4f.Provider import RetryProvider, ChatGLM, DDG, Free2GPT, GizAI, Liaobots, OIVSCode, PollinationsAI
import json
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
from typing import List, Tuple, Dict, Union
from internal.entity_linker.entity_linker import EntityLinker

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

class TermExtractor:
    def __init__(self):
        providers = [RetryProvider, ChatGLM, DDG, Free2GPT, GizAI, Liaobots, OIVSCode, PollinationsAI]
        self.client = Client(provider=RetryProvider(providers, shuffle=False))
        self.model = "gpt-4o-mini"

        self.prompt_template = (
            "Проанализируй приведенный ниже текст на русском языке и извлеки из него термины лекции, характерные для математики и компьютерных наук. "
            "Это должны быть только термины. Перечисли только термины через запятую без каких-либо дополнительных комментариев.\n"
            "Текст: \"{}\""
        )

        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)
    
    def extract_terms_chunk(self, text: str) -> List[str]:
        """Извлекает термины из переданного текста."""
        prompt = self.prompt_template.format(text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            response_content = response.choices[0].message.content
            terms = [term.strip() for term in response_content.split(",") if term.strip()]
            return terms
        except Exception as e:
            print(f"Ошибка при извлечении терминов: {e}")
            return []
    
    def lemmatize_text(self, text: str) -> str:
        """Лемматизирует входной текст, возвращая строку с леммами."""
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        tokens = []
        for token in doc.tokens:
            if token.pos in {'PUNCT', 'SYM'}:
                continue
            token.lemmatize(self.morph_vocab)
            tokens.append(token.lemma.lower() if token.lemma else token.text.lower())
        return ' '.join(tokens)
    
    def split_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Делит текст на части, каждая не длиннее max_chunk_size символов.
        Предпочтительно делит по границам предложений (точкам).
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            # Ищем последнюю точку в пределах текущего отрезка
            period_index = text.rfind(".", start, end)
            if period_index == -1:
                period_index = end  # если точек нет, берём отрезок фиксированной длины
            else:
                period_index += 1  # включаем саму точку
            chunk = text[start:period_index].strip()
            if chunk:
                chunks.append(chunk)
            start = period_index
        return chunks
    def extract_terms(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """
        Делит большой текст на части, параллельно извлекает термины для каждой части, 
        проверяет их наличие в Wikidata и объединяет их, устраняя дублирование.
        """
        chunks = self.split_text(text, max_chunk_size)
        valid_terms = set()
        
        def process_chunk(chunk: str, idx: int) -> List[str]:
            print(f"Обработка части {idx + 1}/{len(chunks)}")
            terms = self.extract_terms_chunk(chunk)
            valid_terms_chunk = []
            for term in terms:
                valid_terms_chunk.append(term)
            return valid_terms_chunk
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, chunk, idx) for idx, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                valid_terms.update(future.result())

        return list(valid_terms)
    def validate_terms(self, text: str, raw_terms: List[str], entity_linker: EntityLinker) -> List[str]:
        valid_terms = []

        def validate_term(term: str) -> bool:
            linked_entities = entity_linker.link_entities(term)
            return bool(linked_entities)

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(validate_term, term): term for term in raw_terms}
            for future in as_completed(futures):
                term = futures[future]
                try:
                    if future.result():
                        valid_terms.append(term)
                except Exception as e:
                    logger.error(f"Error'{term}': {e}")

        return valid_terms

    def extract_sentences_with_terms(self, text: str, terms: List[str]) -> Dict[str, List[str]]:
        """
        Извлекает предложения, в которых присутствуют лемматизированные термины,
        а также соседние предложения (до и после найденного).
        Возвращает словарь, где ключ — термин, значение — список предложений.
        """
        doc = Doc(text)
        doc.segment(self.segmenter)

        terms_lemmatized = {term: self.lemmatize_text(term) for term in terms}
        matching_sentences = {term: [] for term in terms}
        sentences = list(doc.sents)

        for i, sent in enumerate(sentences):
            lemmatized_sentence = self.lemmatize_text(sent.text)
            for term, term_lemmatized in terms_lemmatized.items():
                if term_lemmatized in lemmatized_sentence:
                    if i > 0:
                        matching_sentences[term].append(sentences[i-1].text)
                    matching_sentences[term].append(sent.text)
                    if i < len(sentences) - 1:
                        matching_sentences[term].append(sentences[i+1].text)
                        
        return matching_sentences





