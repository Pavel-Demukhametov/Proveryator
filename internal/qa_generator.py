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
    
    def gen_questions(self, text: str, num_questions: int, question_type: str = 'mc') -> List[str]:
        """
        Генерирует заданное количество вопросов на основе контекста.
        """
        input_text = f"Generate a {question_type} question: {text}"
        input_ids = self.tokenizer(
            input_text,
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
    
    def generate_qa_pairs(self, text: str, num_questions: int = 10, question_type: str = 'mc') -> List[Dict[str, Optional[List[str]]]]:
        """
        Генерирует пары вопрос-ответ с дистракторами на основе заданного текста.
        """
        if question_type not in ['mc', 'open']:
            raise ValueError(f"Неизвестный тип вопроса: {question_type}")

        questions = self.gen_questions(text, num_questions, question_type)
        qa_pairs = []

        for quest in questions:
            answer = self.gen_answer(text, quest)
            if question_type == 'mc':
                distractors = self.gen_distractors(text, answer)
            else:
                distractors = None

            qa_pair = {
                "type": question_type,
                "Вопрос": quest,
                "Ответ": answer,
                "Дистракторы": distractors
            }
            qa_pairs.append(qa_pair)

        return qa_pairs


# internal/qa_generator.py

# from gensim.models import KeyedVectors
# from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
# import nltk
# from nltk.corpus import stopwords
# import torch
# from transformers import AutoTokenizer, T5ForConditionalGeneration
# from typing import List, Optional, Dict

# # Инициализация NLTK ресурсов
# nltk.download('stopwords')


# class QAGenerator:
#     def __init__(
#         self,
#         question_model_path: str = 'question_generation_model',
#         answer_model_path: str = 'final_model',
#         question_from_answer_model_path: str = 'hivaze/AAQG-QA-QG-FRED-T5-large',  # Новый путь к модели
#         tokenizer_model: str = "ai-forever/ruT5-base",
#         answer_doublication: int = 25,
#         distractor_spread: int = 10,
#         fractions: List[float] = [1/3, 2/3, 1.0]
#     ):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
#         self.question_model = T5ForConditionalGeneration.from_pretrained(question_model_path).to(self.device)
#         self.answer_model = T5ForConditionalGeneration.from_pretrained(answer_model_path).to(self.device)
#         self.question_from_answer_model = T5ForConditionalGeneration.from_pretrained(question_from_answer_model_path).to(self.device)  # Инициализация новой модели

#         self.answer_doublication = answer_doublication
#         self.distractor_spread = distractor_spread
#         self.fractions = fractions

#         self.segmenter = Segmenter()
#         self.stop_words = set(stopwords.words('russian'))
#         self.punct = {',', '?', '!', '.', ';', ':', '...', '(', ')', '\'', '\"', '«', '»'}

#         self.emb_model = self._load_embedding_model()

#     def _load_embedding_model(self) -> KeyedVectors:
#         """
#         Конвертирует модель Natasha NewsEmbedding в формат Gensim KeyedVectors.
#         """
#         news_emb = NewsEmbedding()
#         return self._as_gensim(news_emb)

#     @staticmethod
#     def _as_gensim(navec_model) -> KeyedVectors:
#         """
#         Преобразует модель Natasha в формат Gensim KeyedVectors.
#         """
#         model = KeyedVectors(vector_size=navec_model.pq.dim)
#         weights = navec_model.pq.unpack()
#         model.add_vectors(navec_model.vocab.words, weights)
#         return model
    
#     def tokenize(self, text: str) -> List[str]:
#         """
#         Токенизирует текст, используя Natasha.
#         """
#         doc = Doc(text)
#         doc.segment(self.segmenter)
#         return [token.text.lower() for token in doc.tokens]
    
#     def model_words_filter(self, tokens: List[str]) -> List[str]:
#         """
#         Фильтрует токены, оставляя только те, которые присутствуют в модели и не являются стоп-словами или пунктуацией.
#         """
#         return [token for token in tokens if token in self.emb_model and token not in self.stop_words and token not in self.punct]
    
#     def gen_distractors(self, text: str, answer: str) -> Optional[List[Optional[str]]]:
#         """
#         Генерирует дистракторы для данного ответа на основе контекста.
#         """
#         answer = answer.lower()
#         if answer not in self.emb_model:
#             # print(f"Ответа '{answer}' нет в модели")
#             return None

#         tokens = self.tokenize(text)
#         filtered_tokens = self.model_words_filter(tokens)

#         for _ in range(self.answer_doublication):
#             filtered_tokens.append(answer)

#         most_similar_words = self.emb_model.most_similar(positive=filtered_tokens, topn=self.distractor_spread)
#         distractors = []
#         for fraction in self.fractions:
#             index = int(self.distractor_spread * fraction) - 1
#             if 0 <= index < len(most_similar_words):
#                 distractors.append(most_similar_words[index][0])
#             else:
#                 distractors.append(None)

#         return distractors
    
#     def gen_answer(self, text: str, question: str) -> str:
#         """
#         Генерирует ответ на заданный вопрос на основе контекста.
#         """
#         input_text = f"{question} {text}"
        
#         input_ids = self.tokenizer(
#             input_text,
#             return_tensors="pt",
#             max_length=512,
#             truncation=True,
#             padding="max_length",
#         ).input_ids.to(self.device)

#         answer_ids = self.answer_model.generate(
#             input_ids, max_length=128, num_beams=5, early_stopping=True
#         )

#         correct_answer = self.tokenizer.decode(
#             answer_ids[0], skip_special_tokens=True
#         )

#         return correct_answer
    
#     def gen_questions(self, text: str, num_questions: int, question_type: str = 'mc') -> List[str]:
#         """
#         Генерирует заданное количество вопросов на основе контекста.
#         """
#         input_text = f"Generate a {question_type} question: {text}"
#         input_ids = self.tokenizer(
#             input_text,
#             return_tensors="pt",
#             max_length=512,
#             truncation=True,
#             padding="max_length",
#         ).input_ids.to(self.device)

#         question_ids = self.question_model.generate(
#             input_ids,
#             max_length=64,
#             num_beams=12,
#             num_return_sequences=num_questions,
#             temperature=1.2,
#             top_p=0.9,
#             early_stopping=True,
#         )

#         questions = [
#             self.tokenizer.decode(q, skip_special_tokens=True) for q in question_ids
#         ]

#         return questions


#     def generate_question_from_answer(self, context: str, answer: str, question_type: str = 'open', n: int = 1, temperature: float = 0.8, num_beams: int = 3) -> List[str]:
#         """
#         Генерирует вопрос на основе известного ответа и контекста.

#         :param context: Текстовый контекст.
#         :param answer: Известный ответ.
#         :param question_type: Тип вопроса ('open' или 'mc').
#         :param n: Количество вариантов вопросов для генерации.
#         :param temperature: Температура генерации.
#         :param num_beams: Количество лучей для генерации.
#         :return: Список сгенерированных вопросов.
#         """
#         if question_type not in ['open', 'mc']:
#             raise ValueError("Неподдерживаемый тип вопроса. Используйте 'open' или 'mc'.")

#         if question_type == 'open':
#             prompt = f"Сгенерируй вопрос по тексту, используя известный ответ. Текст: '{context}'. Ответ: '{answer}'."
#         else:  # 'mc'
#             prompt = f"Сгенерируй вопрос по тексту, используя известный ответ. Текст: '{context}'. Ответ: '{answer}'."

#         input_ids = self.tokenizer.encode_plus(
#             prompt,
#             return_tensors='pt',
#             max_length=512,
#             truncation=True,
#             padding="max_length"
#         ).input_ids.to(self.device)

#         question_ids = self.question_from_answer_model.generate(
#             input_ids,
#             max_length=64,
#             num_beams=num_beams,
#             num_return_sequences=n,
#             temperature=temperature,
#             top_p=0.9,
#             top_k=50,
#             do_sample=True,
#             early_stopping=True
#         )

#         questions = [
#             self.tokenizer.decode(q, skip_special_tokens=True) for q in question_ids
#         ]

#         return questions

#     def gen_questions_from_answer(self, text: str, answer: str, num_questions: int, question_type: str = 'mc') -> List[str]:
#         """
#         Генерирует заданное количество вопросов на основе известного ответа и контекста.

#         :param text: Текстовый контекст.
#         :param answer: Известный ответ.
#         :param num_questions: Количество вопросов для генерации.
#         :param question_type: Тип вопроса ('mc' или 'open').
#         :return: Список сгенерированных вопросов.
#         """
#         return self.generate_question_from_answer(
#             context=text,
#             answer=answer,
#             question_type=question_type,
#             n=num_questions
#         )

        
#     def generate_qa_pairs(self, text: str, num_questions: int = 10, question_type: str = 'mc') -> List[Dict[str, Optional[List[str]]]]:
#         """
#         Генерирует пары вопрос-ответ с дистракторами на основе заданного текста.
#         """
#         if question_type not in ['mc', 'open']:
#             raise ValueError(f"Неизвестный тип вопроса: {question_type}")

#         questions = self.gen_questions(text, num_questions, question_type)
#         qa_pairs = []

#         for quest in questions:
#             answer = self.gen_answer(text, quest)
#             if question_type == 'mc':
#                 distractors = self.gen_distractors(text, answer)
#             else:
#                 distractors = None

#             qa_pair = {
#                 "type": question_type,
#                 "Вопрос": quest,
#                 "Ответ": answer,
#                 "Дистракторы": distractors
#             }
#             qa_pairs.append(qa_pair)

#         return qa_pairs

#     def generate_qa_pairs_from_answer(self, text: str, answer: str, num_questions: int = 5, question_type: str = 'mc') -> List[Dict[str, Optional[List[str]]]]:
#         """
#         Генерирует пары вопрос-ответ с дистракторами на основе известного ответа и текста.

#         :param text: Текстовый контекст.
#         :param answer: Известный ответ.
#         :param num_questions: Количество вопросов для генерации.
#         :param question_type: Тип вопроса ('mc' или 'open').
#         :return: Список сгенерированных пар вопрос-ответ.
#         """
#         questions = self.gen_questions_from_answer(text, answer, num_questions, question_type)
#         qa_pairs = []

#         for quest in questions:
#             if question_type == 'mc':
#                 distractors = self.gen_distractors(text, answer)
#             else:
#                 distractors = None

#             qa_pair = {
#                 "type": question_type,
#                 "Вопрос": quest,
#                 "Ответ": answer,
#                 "Дистракторы": distractors
#             }
#             qa_pairs.append(qa_pair)

#         return qa_pairs
