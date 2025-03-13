# import torch
# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger

# class MBartTermExtractor:
#     def __init__(self):
#         self.tokenizer = MBart50TokenizerFast.from_pretrained(
#             "facebook/mbart-large-50", 
#             src_lang="ru_RU", 
#             tgt_lang="ru_RU"
#         )
#         self.model = MBartForConditionalGeneration.from_pretrained(r"D:\Program Files\Lecture_test_front\fast\internal\term_extractor\mbart-1000")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()
        
#         self.segmenter = Segmenter()
#         self.emb = NewsEmbedding()
#         self.morph_vocab = MorphVocab()
#         self.morph_tagger = NewsMorphTagger(self.emb)
    
#     def extract_terms(self, text: str) -> list:
#         input_text = f"Выдели термины, характерные для математики и компьютерных наук: {text}"
#         inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             output_tokens = self.model.generate(**inputs, max_length=128)
#         extracted_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
#         return [term.strip() for term in extracted_text.split(",")]
#   
#    def extract_sentences_with_terms(self, text: str, terms: list) -> dict:
#         """
#         Для каждого извлечённого термина ищет предложения, в которых присутствует лемматизированный термин.
#         Поиск производится по лемматизированному содержимому предложения.
#         """
#         doc = Doc(text)
#         doc.segment(self.segmenter)
#         # Для каждого термина сначала получаем его лемматизированную форму
#         results = {}
#         for term in terms:
#             term_lemmatized = self.lemmatize_text(term)
#             matching_sentences = []
#             for sent in doc.sents:
#                 lemmatized_sentence = self.lemmatize_text(sent.text)
#                 # Если лемма термина содержится в лемматизированном предложении, сохраняем оригинальное предложение
#                 if term_lemmatized in lemmatized_sentence:
#                     matching_sentences.append(sent.text)
#             results[term] = matching_sentences
#         return results


from typing import List, Dict
from g4f.client import Client
from g4f.Provider import RetryProvider, ChatGLM, DDG, Free2GPT, GizAI, Liaobots, OIVSCode, PollinationsAI
import json
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger

class MBartTermExtractor:
    def __init__(self):
        providers = [DDG, PollinationsAI, Liaobots, OIVSCode, Free2GPT, GizAI, ChatGLM]
        self.client = Client(provider=RetryProvider(providers, shuffle=False))
        self.model = "gpt-4o-mini"

        self.prompt_template = (
            "Проанализируй приведенный ниже текст на русском языке и извлеки из него ключевые термины лекции, характерные для математики и компьютерных наук. "
            "Перечисли только термины через запятую без каких-либо дополнительных комментариев.\n"
            "Текст: \"{}\""
        )

        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)
    
    def extract_terms(self, text: str) -> list:
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
    def extract_sentences_with_term(self, text: str, term: str) -> list:
        """
        Извлекает предложения, в которых присутствует лемматизированный термин,
        а также соседние предложения (до и после найденного).
        Возвращает список предложений.
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        term_lemmatized = self.lemmatize_text(term)
        matching_sentences = []
        sentences = list(doc.sents)
        
        for i, sent in enumerate(sentences):
            lemmatized_sentence = self.lemmatize_text(sent.text)
            if term_lemmatized in lemmatized_sentence:
                if i > 0:
                    matching_sentences.append(sentences[i-1].text)
                matching_sentences.append(sent.text)
                if i < len(sentences) - 1:
                    matching_sentences.append(sentences[i+1].text)
                    
        return matching_sentences

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