import re
from typing import List, Tuple, Dict
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.corpus import stopwords
import spacy
import nltk
# Инициализация инструментов
nlp = spacy.load('ru_core_news_sm')
stop_words = set(stopwords.words('russian'))

# Удаление дубликатов предложений
def remove_duplicate_sentences(sentences):
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        normalized = sentence.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)
    return unique_sentences

# Проверка, является ли предложение шумным
def is_noise(s: str) -> bool:
    doc = nlp(s)
    if not any(tok.pos_ == 'VERB' for tok in doc):
        return True
    words = [tok.text for tok in doc if tok.is_alpha]
    if words and sum(len(w) <= 2 for w in words) / len(words) > 0.3:
        return True
    if re.search(r'(www\.|https?://|\.com|\.ru)', s):
        return True
    if re.search(r'\b[А-ЯЁ]{4,}\b', s):
        return True
    return False

# Фильтрация предложений по наличию ключевых слов 
def filter_sentences(sentences: List[str], keywords: List[str]) -> List[Tuple[int, str]]:
    filtered = []
    for i, s in enumerate(sentences):
        if any(keyword.lower() in s.lower() for keyword in keywords) and not is_noise(s):
            filtered.append((i, s))
    return filtered

# Разделение больших предложений
def split_into_sentences(text: str, min_w=2, max_w=30) -> list:
    raw = nltk.sent_tokenize(text)
    return [s.strip() for s in raw if min_w <= len(s.split()) <= max_w and any(c.isalpha() for c in s)]

# Вычисление LexRank-оценок
def compute_lexrank(sentences: List[str]) -> Dict[str, float]:
    parser = PlaintextParser.from_string(' '.join(sentences), Tokenizer('russian'))
    summarizer = LexRankSummarizer()
    summarizer.stop_words = stop_words
    summary = summarizer(parser.document, len(sentences))
    n = len(sentences)
    return {str(s): (n - i) / n for i, s in enumerate(summary)}

# Вычисление оценок новизны
def compute_novelty_scores(sentences: List[str]) -> Dict[str, float]:
    seen = set()
    nov = {}
    for sent in sentences:
        words = set(sent.split())
        new = words - seen
        nov[sent] = len(new)
        seen |= words
    if not nov:
        return {}
    maxv = max(nov.values())
    return {s: v / maxv for s, v in nov.items()}

# Ранжирование предложений по LexRank и новизне
def rank_sentences_globally(indexed_sentences: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    sentences = [s for _, s in indexed_sentences]
    lex_scores = compute_lexrank(sentences)
    novelty_scores = compute_novelty_scores(sentences)
    
    def sort_key(item: Tuple[int, str]) -> Tuple[float, float]:
        idx, s = item
        return (-lex_scores.get(s, 0), -novelty_scores.get(s, 0))
    
    return sorted(indexed_sentences, key=sort_key)

def rank_sentences_with_index(sentences: List[str], keywords: List[str]) -> List[Tuple[int, str]]:
    """
    Ранжирует предложения, фильтруя их по наличию ключевых слов и шуму, а затем сортируя по LexRank и новизне.
        
    Возвращает:
        Список кортежей (индекс, предложение), отсортированных по убыванию релевантности,
        где релевантность определяется LexRank-оценкой и новизной.
    """
    filtered = filter_sentences(sentences, keywords)
    if not filtered:
        return []
    ranked = rank_sentences_globally(filtered)
    return ranked



# Добавление контекста к предложениям
def build_context_window(sentences: List[str], center_index: int) -> str:
    window = sentences[max(0, center_index - 1): center_index + 2]
    return " ".join(window)
