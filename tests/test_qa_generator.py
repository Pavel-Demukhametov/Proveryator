# test_internal.qa_generator.py

import pytest
from unittest.mock import patch, MagicMock
from internal.qa_generator import (
    get_access_token,
    parse_distractors,
    generate_distractors,
    QAGenerator
)
from unittest.mock import patch, MagicMock, ANY
from gensim.models import KeyedVectors
# Тест для функции get_access_token
@patch('internal.qa_generator.requests.post')
def test_get_access_token(mock_post):
    # Настройка мок-ответа
    mock_response = MagicMock()
    mock_response.json.return_value = {'access_token': 'test_token'}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    token = get_access_token()
    assert token == 'test_token'
    mock_post.assert_called_once_with(
        'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': '639a832c-d12a-4c2f-9207-f654e0d86fa6',
            'Authorization': 'Basic NjM5YTgzMmMtZDEyYS00YzJmLTkyMDctZjY1NGUwZDg2ZmE2OmE4MTY2MTg1LWQ1OWEtNGVkOC05ZmEzLWRmYjRlOTdjNDAxOA=='
        },
        data='scope=GIGACHAT_API_PERS',
        verify="C:\\Users\\purit\\Downloads\\russian_trusted_root_ca.cer"
    )

# Тесты для функции parse_distractors
@pytest.mark.parametrize("input_str,expected", [
    (
        "1. Дистрактор A; 2. Дистрактор B; 3. Дистрактор C",
        ["Дистрактор A", "Дистрактор B", "Дистрактор C"]
    ),
    (
        "Дистрактор A; Дистрактор B; Дистрактор C",
        ["Дистрактор A", "Дистрактор B", "Дистрактор C"]
    ),
    (
        "Неправильные варианты ответа: 1) Дистрактор A; 2) Дистрактор B; 3) Дистрактор C",
        ["Дистрактор A", "Дистрактор B", "Дистрактор C"]
    ),
    (
        "Дистрактор A. Дистрактор B. Дистрактор C.",
        ["Дистрактор A", "Дистрактор B", "Дистрактор C"]
    ),
    (
        "Дистрактор A; Дистрактор B",
        []  # Менее 3 дистракторов
    ),
    (
        "",
        []  # Пустая строка
    ),
])
def test_parse_distractors(input_str, expected):
    from internal.qa_generator import parse_distractors
    assert parse_distractors(input_str) == expected



# Тестирование класса QAGenerator
@patch('internal.qa_generator.get_access_token')
@patch('internal.qa_generator.T5ForConditionalGeneration.from_pretrained')
@patch('internal.qa_generator.AutoTokenizer.from_pretrained')
@patch('internal.qa_generator.NewsEmbedding')
def test_qagenerator_initialization(mock_news_embedding, mock_tokenizer, mock_t5_from_pretrained, mock_get_access_token):
    # Настройка моков
    mock_get_access_token.return_value = 'test_token'
    mock_tokenizer.return_value = MagicMock()
    mock_t5_from_pretrained.return_value = MagicMock()
    mock_news_embedding.return_value.pq.dim = 300
    mock_news_embedding.return_value.pq.unpack.return_value = [[0.1]*300]*100  # Пример весов
    mock_news_embedding.return_value.vocab.words = [f"word{i}" for i in range(100)]

    qag = QAGenerator()

    assert qag.access_token == 'test_token'
    assert qag.tokenizer is not None
    assert qag.question_model is not None
    assert qag.answer_model is not None
    assert isinstance(qag.emb_model, KeyedVectors)
    assert len(qag.emb_model) == 100

# Тест для метода parse_distractors внутри класса (опционально, если доступно)
# Альтернативно, мы уже протестировали parse_distractors как отдельную функцию

# Тест для метода gen_distractors класса QAGenerator
@patch('internal.qa_generator.generate_distractors')
@patch('internal.qa_generator.get_access_token')
def test_gen_distractors_class(mock_get_access_token, mock_generate_distractors):
    mock_get_access_token.return_value = 'test_token'
    mock_generate_distractors.return_value = ["Дистрактор A", "Дистрактор B", "Дистрактор C"]

    qag = QAGenerator()
    distractors = qag.gen_distractors("Пример текста", "Правильный ответ")
    assert distractors == ["Дистрактор A", "Дистрактор B", "Дистрактор C"]
    mock_generate_distractors.assert_called_once_with('test_token', "Пример текста", "Правильный ответ")
