from gensim.models import KeyedVectors
from natasha import Doc, Segmenter, NewsEmbedding
import nltk
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import List, Optional, Dict
from scipy.spatial.distance import cosine
import numpy as np
import logging
import re
nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import requests
import json

oauth_url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
models_url = 'https://gigachat.devices.sberbank.ru/api/v1/models'
completions_url = 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions'
uid = '639a832c-d12a-4c2f-9207-f654e0d86fa6'
client_base64 = 'NjM5YTgzMmMtZDEyYS00YzJmLTkyMDctZjY1NGUwZDg2ZmE2OmE4MTY2MTg1LWQ1OWEtNGVkOC05ZmEzLWRmYjRlOTdjNDAxOA=='

prompt = (
    'На основе предоставленного текста и правильного ответа, составь 3 неправильных варианта ответа (дистракторы), не надо добавлять ещё какой-то текст или нумеровать неправильные варианты. Твой ответ должен содеражать только 3 неправильных варианта ответа '
    'разделённых точкой с запятой. Текст: {text}. '
    'Правильный ответ: {correct_answer}'
)

def get_access_token():
    """ Получение токена доступа OAuth """
    headers = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Accept': 'application/json',
      'RqUID': uid,
      'Authorization': 'Basic ' + client_base64
    }
    payload = 'scope=GIGACHAT_API_PERS'

    response = requests.post(oauth_url, headers=headers, data=payload, verify="C:\\Users\\purit\\Downloads\\russian_trusted_root_ca.cer")
    return response.json().get('access_token')

def parse_distractors(response_content: str) -> List[str]:
    """
    Парсит дистракторы из строки, учитывая различные форматы.
    Поддерживает разделение точкой с запятой и/или нумерацию.
    """
    distractors = []

    response_content = re.sub(r'Неправильные варианты ответа[:\-]?\s*', '', response_content, flags=re.IGNORECASE)
    response_content = re.sub(r'3 неправильных варианта ответа[:\-]?\s*', '', response_content, flags=re.IGNORECASE)

    # Измененный шаблон для обработки нумерации с точкой и скобкой
    numbered_distractors = re.findall(r'\d+[.)]\s*([^;]+)', response_content)
    if numbered_distractors and len(numbered_distractors) >= 3:
        distractors = [distractor.strip() for distractor in numbered_distractors[:3]]
        logger.info("Дистракторы успешно распознаны с нумерацией.")
        return distractors

    split_distractors = [distractor.strip() for distractor in response_content.split(';') if distractor.strip()]
    if len(split_distractors) >= 3:
        distractors = split_distractors[:3]
        logger.info("Дистракторы успешно распознаны без нумерации.")
        return distractors

    # Измененный шаблон для разделения по нумерации с точкой и скобкой
    additional_split = re.split(r'\d+[.)]\s*', response_content)
    additional_split = [distractor.strip() for distractor in additional_split if distractor.strip()]
    if len(additional_split) >= 3:
        distractors = additional_split[:3]
        logger.info("Дистракторы успешно распознаны при разделении по нумерации без точек с запятой.")
        return distractors

    sentences = re.split(r'[.!?]\s*', response_content)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(sentences) >= 3:
        distractors = sentences[:3]
        logger.warning("Дистракторы распознаны по предложениям, так как другие методы не сработали.")
        return distractors
    logger.error("Не удалось распознать дистракторы из ответа.")
    return distractors

def generate_distractors(access_token: str, text: str, correct_answer: str) -> List[str]:
    """Генерация дистракторов на основе текста и правильного ответа"""
    if not access_token:
        logger.error("Отсутствует токен доступа OAuth.")
        return []

    formatted_prompt = prompt.format(text=text, correct_answer=correct_answer)
    
    payload = json.dumps({
        "model": "GigaChat",
        "messages": [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "stream": False,
        "repetition_penalty": 1
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + access_token
    }

    try:
        response = requests.post(
            completions_url,
            headers=headers,
            data=payload,
            verify="C:\\Users\\purit\\Downloads\\russian_trusted_root_ca.cer"
        )
        response.raise_for_status()
        response_json = response.json()
        logger.info("Ответ от GigaChat получен успешно.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при обращении к GigaChat API: {e}")
        return []

    if "choices" not in response_json or not response_json["choices"]:
        logger.error("Ответ от GigaChat не содержит необходимых данных.")
        return []

    response_content = response_json["choices"][0]["message"]["content"]
    logger.debug(f"Содержимое ответа от GigaChat: {response_content}")

    distractors = parse_distractors(response_content)

    if len(distractors) < 3:
        additional_distractors = [distractor.strip() for distractor in response_content.split(';') if distractor.strip()]
        for distractor in additional_distractors:
            if distractor not in distractors:
                distractors.append(distractor)
            if len(distractors) == 3:
                break

    if len(distractors) < 3:
        return []
    
    logger.info(f"Дистракторы успешно сгенерированы: {distractors}")
    return distractors



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
        self.access_token = get_access_token()
        
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
    
    def get_sentence_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Вычисляет векторное представление предложения путем усреднения векторов слов.
        """
        tokens = self.tokenize(text)
        filtered_tokens = self.model_words_filter(tokens)
        if not filtered_tokens:
            return None
        vectors = [self.emb_model[token] for token in filtered_tokens if token in self.emb_model]
        if not vectors:
            return None
        embedding = np.mean(vectors, axis=0)
        return embedding

    def is_similar(self, existing_embeddings: List[np.ndarray], new_embedding: np.ndarray, threshold: float = 0.8) -> bool:
        """
        Проверяет, схож ли новый вектор с уже существующими.
        """
        for emb in existing_embeddings:
            similarity = 1 - cosine(emb, new_embedding)
            if similarity >= threshold:
                return True
        return False
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
    def gen_distractors(self, text: str, answer: str):
        """
        Генерация дистракторов через GigaChat API.
        """
        result = generate_distractors(self.access_token, text, answer)
        if isinstance(result, list):
            return result
        else:
            return []
    
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
