# internal/qa_generator.py

from gensim.models import KeyedVectors
from natasha import Doc, Segmenter, NewsEmbedding
import nltk
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import List, Optional, Dict

# Инициализация NLTK ресурсов
nltk.download('stopwords')


class QAGenerator:
    def __init__(
        self,
        question_model_path: str = 'question_generation_model',
        answer_model_path: str = 'final_model',
        tokenizer_model: str = "ai-forever/ruT5-base",
        answer_doublication: int = 25,
        distractor_spread: int = 10,
        fractions: List[float] = [1/3, 2/3, 1.0]
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.question_model = T5ForConditionalGeneration.from_pretrained(question_model_path).to(self.device)
        self.answer_model = T5ForConditionalGeneration.from_pretrained(answer_model_path).to(self.device)
        
        self.answer_doublication = answer_doublication
        self.distractor_spread = distractor_spread
        self.fractions = fractions
        
        self.segmenter = Segmenter()
        self.stop_words = set(stopwords.words('russian'))
        self.punct = {',', '?', '!', '.', ';', ':', '...', '(', ')', '\'', '\"', '«', '»'}
        
        self.emb_model = self._load_embedding_model()
    
    def _load_embedding_model(self) -> KeyedVectors:
        """
        Конвертирует модель Natasha NewsEmbedding в формат Gensim KeyedVectors.
        """
        news_emb = NewsEmbedding()
        return self._as_gensim(news_emb)
    
    @staticmethod
    def _as_gensim(navec_model) -> KeyedVectors:
        """
        Преобразует модель Natasha в формат Gensim KeyedVectors.
        """
        model = KeyedVectors(vector_size=navec_model.pq.dim)
        weights = navec_model.pq.unpack()
        model.add_vectors(navec_model.vocab.words, weights)
        return model
    
    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст, используя Natasha.
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        return [token.text.lower() for token in doc.tokens]
    
    def model_words_filter(self, tokens: List[str]) -> List[str]:
        """
        Фильтрует токены, оставляя только те, которые присутствуют в модели и не являются стоп-словами или пунктуацией.
        """
        return [token for token in tokens if token in self.emb_model and token not in self.stop_words and token not in self.punct]
    
    def gen_distractors(self, text: str, answer: str) -> Optional[List[Optional[str]]]:
        """
        Генерирует дистракторы для данного ответа на основе контекста.
        """
        answer = answer.lower()
        if answer not in self.emb_model:
            # print(f"Ответа '{answer}' нет в модели")
            return None

        tokens = self.tokenize(text)
        filtered_tokens = self.model_words_filter(tokens)

        for _ in range(self.answer_doublication):
            filtered_tokens.append(answer)

        most_similar_words = self.emb_model.most_similar(positive=filtered_tokens, topn=self.distractor_spread)
        distractors = []
        for fraction in self.fractions:
            index = int(self.distractor_spread * fraction) - 1
            if 0 <= index < len(most_similar_words):
                distractors.append(most_similar_words[index][0])
            else:
                distractors.append(None)

        return distractors
    
    def gen_answer(self, text: str, question: str) -> str:
        """
        Генерирует ответ на заданный вопрос на основе контекста.
        """
        input_text = f"{question} {text}"
        
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).input_ids.to(self.device)

        answer_ids = self.answer_model.generate(
            input_ids, max_length=128, num_beams=5, early_stopping=True
        )

        correct_answer = self.tokenizer.decode(
            answer_ids[0], skip_special_tokens=True
        )

        return correct_answer
    
    def gen_questions(self, text: str, num_questions: int) -> List[str]:
        """
        Генерирует заданное количество вопросов на основе контекста.
        """
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).input_ids.to(self.device)

        question_ids = self.question_model.generate(
            input_ids,
            max_length=64,
            num_beams=12,
            num_return_sequences=num_questions,
            temperature=1.2,
            top_p=0.9,
            early_stopping=True,
        )

        questions = [
            self.tokenizer.decode(q, skip_special_tokens=True) for q in question_ids
        ]

        return questions
    
    def generate_qa_pairs(self, text: str, num_questions: int = 10) -> List[Dict[str, Optional[List[str]]]]:
        """
        Генерирует пары вопрос-ответ с дистракторами на основе заданного текста.
        """
        questions = self.gen_questions(text, num_questions)
        qa_pairs = []

        for quest in questions:
            answer = self.gen_answer(text, quest)
            distractors = self.gen_distractors(text, answer)

            qa_pair = {
                "Вопрос": quest,
                "Ответ": answer,
                "Дистракторы": distractors
            }
            qa_pairs.append(qa_pair)

        return qa_pairs
