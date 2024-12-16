import pytest
from internal.keywords_extractor import KeywordsExtractor

@pytest.fixture
def extractor():
    return KeywordsExtractor(top_n=5)

@pytest.fixture
def sample_text():
    return (
        "Программирование на Python позволяет создавать различные приложения. "
        "Python прост в изучении и обладает богатой стандартной библиотекой. "
        "Множество разработчиков используют Python для веб-разработки, анализа данных и автоматизации."
    )

def test_extract_keywords_default(extractor, sample_text):
    keywords = extractor.extract_keywords(sample_text)
    assert isinstance(keywords, list)
    assert len(keywords) <= 5
    for keyword in keywords:
        assert isinstance(keyword, str)

def test_extract_keywords_custom_top_n(extractor, sample_text):
    keywords = extractor.extract_keywords(sample_text, top_n=3)
    assert len(keywords) <= 3

def test_tokenize_lemmatize(extractor):
    text = "Программирование на Python — это весело."
    lemmatized = extractor.tokenize_lemmatize(text)
    expected = "программирование на python это веселый"
    assert lemmatized.lower() == expected.lower()

def test_extract_sentences_with_keyword(extractor, sample_text):
    keyword = "python"
    lemmatized_keyword = extractor.tokenize_lemmatize(keyword)
    sentences = extractor.extract_sentences_with_keyword(sample_text, lemmatized_keyword)
    assert isinstance(sentences, list)
    assert len(sentences) >= 1
    for sentence in sentences:
        assert "Python" in sentence

def test_extract_keywords_empty_text(extractor):
    keywords = extractor.extract_keywords("")
    assert keywords == []

def test_extract_sentences_with_keyword_no_match(extractor, sample_text):
    keyword = "Java"
    lemmatized_keyword = extractor.tokenize_lemmatize(keyword)
    sentences = extractor.extract_sentences_with_keyword(sample_text, lemmatized_keyword)
    assert sentences == []
