# internal/test_generator.py

from transformers import AutoTokenizer, T5ForConditionalGeneration
from rutermextract import TermExtractor
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
import torch
import nltk
from nltk.corpus import stopwords
from typing import List

# Инициализация NLTK ресурсов
nltk.download('stopwords')

class TestGenerator:
    def __init__(self, model_name='hivaze/AAQG-QA-QG-FRED-T5-large'):
        self.tokenizer, self.model, self.device = self.load_model_and_tokenizer(model_name)
        self.term_extractor = TermExtractor()
        self.stop_words = set(stopwords.words('russian'))
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)
        # Шаблоны для различных типов вопросов
        self.AAQG_PROMPT_OPEN = "Сгенерируй открытый вопрос по тексту, используя известный ответ. Текст: '{context}'. Ответ: '{answer}'."
        self.AAQG_PROMPT_MC = "Сгенерируй вопрос с выбором из нескольких вариантов по тексту, используя известный ответ. Текст: '{context}'. Ответ: '{answer}'."

    def load_model_and_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        return tokenizer, model, device

    def extract_keywords(self, text, top_n=10):
        keywords = self.term_extractor(text)[:top_n]
        return keywords

    def tokenize_lemmatize(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            if token.lemma:
                token.text = token.lemma

        return ' '.join(token.text for token in doc.tokens)

    def extract_sentences_with_keyword(self, text: str, keyword_lemmatized: str):
        """
        Извлекает предложения из текста, содержащие заданное лемматизированное ключевое слово.
        
        :param text: Полный текст лекции.
        :param keyword_lemmatized: Лемматизированное ключевое слово для поиска.
        :return: Список предложений, содержащих ключевое слово.
        """
        sentences_with_keyword = []
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for sent in doc.sents:
            lemmatized_sent = self.tokenize_lemmatize(sent.text)
            if keyword_lemmatized in lemmatized_sent:
                sentences_with_keyword.append(sent.text)
        return sentences_with_keyword

    def generate_question(self, context, answer, question_type='open', n=1, temperature=0.8, num_beams=3):
        if question_type == 'open':
            prompt = self.AAQG_PROMPT_OPEN.format(context=context, answer=answer)
        elif question_type == 'mc':
            prompt = self.AAQG_PROMPT_MC.format(context=context, answer=answer)
        else:
            raise ValueError("Неподдерживаемый тип вопроса. Используйте 'open' или 'mc'.")

        encoded_input = self.tokenizer.encode_plus(prompt, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        resulted_tokens = self.model.generate(
            **encoded_input,
            max_new_tokens=64,
            do_sample=True,
            num_beams=num_beams,
            num_return_sequences=n,
            temperature=temperature,
            top_p=0.9,
            top_k=50
        )

        resulted_texts = self.tokenizer.batch_decode(resulted_tokens, skip_special_tokens=True)
        return resulted_texts

    def process_text_by_theme(self, themes: List[dict]):
        questions = []

        for theme in themes:
            keyword = theme['keyword']
            sentences = theme['sentences']
            multiple_choice_count = theme.get('multipleChoiceCount', 0)
            open_answer_count = theme.get('openAnswerCount', 0)

            # Объединяем все предложения в один контекст
            combined_sentences = ' '.join(sentences)

            # Генерация открытых вопросов
            generated_questions_open = self.generate_question(
                context=combined_sentences,
                answer=keyword,
                question_type='open',
                n=open_answer_count
            )
            for question in generated_questions_open:
                questions.append({
                    "type": "open",
                    "answer": keyword,
                    "sentence": combined_sentences,
                    "question": question
                })

            # Генерация вопросов с выбором из нескольких вариантов
            generated_questions_mc = self.generate_question(
                context=combined_sentences,
                answer=keyword,
                question_type='mc',
                n=multiple_choice_count
            )
            for question in generated_questions_mc:
                questions.append({
                    "type": "mc",
                    "answer": keyword,
                    "sentence": combined_sentences,
                    "question": question
                    # Можно добавить поле "options": [...]
                })

        return questions