import asyncio
import re
from typing import List, Tuple, Dict, Union
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
from razdel import sentenize, tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")
# Инициализация необходимых инструментов
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

VOWELS = "аеёиоуыэюя"
def remove_duplicate_sentences(sentences):
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        normalized = sentence.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)
    return unique_sentences
def split_into_sentences(text: str) -> List[str]:
    """
    Разбивает текст на предложения с использованием razdel.
    """
    return [s.text for s in sentenize(text)]

def count_syllables_ru(word: str) -> int:
    """
    Подсчёт слогов: каждая гласная считается за один слог.
    """
    return sum(ch in VOWELS for ch in word.lower())

def calculate_flesch_rus(sentence: str) -> float:
    """
    Рассчитывает индекс удобочитаемости Флеша для одного предложения.
    """
    words = list(tokenize(sentence))
    num_words = len(words)
    if num_words == 0:
        return 0.0
    total_syllables = sum(count_syllables_ru(w.text) for w in words)
    asw = total_syllables / num_words
    asl = num_words
    return 206.835 - 1.52 * asl - 65.14 * asw

def calculate_usefulness_for_keywords(sentence: str, keywords: List[str]) -> float:
    """
    Подсчитывает полезность предложения по входу всех ключевых слов с использованием стемминга.
    Для этого предложение и ключевые слова приводятся к их стемам.
    """
    # Применяем стемминг к токенам предложения
    words = [stemmer.stem(token.text.lower()) for token in tokenize(sentence)]
    if not words:
        return 0.0
    # Применяем стемминг к ключевым словам
    stemmed_keywords = [stemmer.stem(kw.lower()) for kw in keywords]
    total_score = sum(words.count(kw) / len(words) for kw in stemmed_keywords)
    return total_score

def rank_sentences_globally(sentences: List[str], all_keywords: List[str]) -> List[Tuple[int, str, float, float]]:
    """
    Ранжирует предложения по полезности (на основе ключевых слов) и удобочитаемости.
    Возвращает список кортежей (индекс, предложение, оценка полезности, индекс Флеша).
    """
    indexed = []
    for i, sent in enumerate(sentences):
        usefulness = calculate_usefulness_for_keywords(sent, all_keywords)
        flesch_score = calculate_flesch_rus(sent)
        indexed.append((i, sent, usefulness, flesch_score))
    indexed.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return indexed

def build_context_window(sentences: List[str], center_index: int) -> str:
    """
    Формирует контекстное окно из текущего, предыдущего и следующего предложения.
    """
    window = sentences[max(0, center_index - 1): center_index + 2]
    return " ".join(window)

def sentence_contains_keyword(sentence: str, keyword: str) -> bool:
    """
    Проверяет наличие ключевого слова в предложении.
    """
    return keyword.lower() in sentence.lower()

def rank_sentences_with_index(sentences: List[str], keywords: List[str]) -> List[Tuple[int, str]]:
    """
    Ранжирует предложения по полезности ключевых слов и удобочитаемости.
    
    Аргументы:
        sentences: Список предложений исходного текста.
        keywords: Массив ключевых слов.
        
    Возвращает:
        Список кортежей (индекс, предложение), отсортированных по убыванию релевантности, 
        где релевантность определяется суммарной оценкой полезности (на основе вхождений ключевых слов)
        и индекса удобочитаемости Флеша.
    """
    # Получаем ранжированный список с оценками полезности и удобочитаемости
    ranked = rank_sentences_globally(sentences, keywords)
    # Извлекаем только индекс и предложение
    return [(idx, sent) for idx, sent, usefulness, flesch in ranked]