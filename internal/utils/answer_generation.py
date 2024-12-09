# internal/utils/answer_generation.py

import random
from typing import List
from gensim.models import KeyedVectors
from natasha import Doc, Segmenter, NewsEmbedding
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

answer_doublication = 25
distractor_spread = 10
fractions = [1/3, 2/3, 3/3]
segmenter = Segmenter()
stop_words = set(stopwords.words('russian'))
punct = {',', '?', '!', '.', ';', ':', '...', '(', ')', '\'', '\"', '«', '»'}

def tokenize(text):
    doc = Doc(text)
    doc.segment(segmenter)
    return [token.text.lower() for token in doc.tokens]

def model_words_filter(model, tokens):
    return [token for token in tokens if token in model and token not in stop_words and token not in punct]

def as_gensim(navec_model):
    model = KeyedVectors(navec_model.pq.dim)
    weights = navec_model.pq.unpack()
    model.add_vectors(navec_model.vocab.words, weights)
    return model

# Загружаем модель эмбеддингов один раз
emb_model = as_gensim(NewsEmbedding())

def gen_distractors(emb_model, text, answer):
    answer = answer.lower()
    if answer not in emb_model:
        return None

    text_tokens = tokenize(text)
    text_tokens = model_words_filter(emb_model, text_tokens)

    # Дублируем правильный ответ, чтобы "подтянуть" похожие слова
    for _ in range(answer_doublication):
        text_tokens.append(answer)

    most_similar_words = emb_model.most_similar(positive=text_tokens, topn=distractor_spread)
    distractors = []
    for fraction in fractions:
        index = int(distractor_spread * fraction) - 1 
        if index >= 0 and index < len(most_similar_words):
            distractors.append(most_similar_words[index][0])
        else:
            distractors.append(None) 

    # Фильтруем None, если вдруг попали
    distractors = [d for d in distractors if d is not None]

    # Если не удалось получить нужное количество, вернём None
    if len(distractors) < 3:
        return None

    return distractors

def generate_incorrect_answers(correct_answer: str, lecture_text: str, num_incorrect: int = 3) -> List[str]:
    """
    Генерирует список неверных вариантов ответов на основе семантических связей с учетом корпуса текста.
    Если корректных дистракторов найти не удалось, возвращаем простые заглушки.
    """
    distractors = gen_distractors(emb_model, lecture_text, correct_answer)

    if distractors is None or len(distractors) < num_incorrect:
        # Если не удалось получить дистракторы, используем заглушки
        incorrect_answers = [f"Неверный ответ {i}" for i in range(1, num_incorrect + 1)]
    else:
        # Берём нужное количество дистракторов
        incorrect_answers = distractors[:num_incorrect]

    all_answers = [correct_answer] + incorrect_answers
    random.shuffle(all_answers)
    return all_answers