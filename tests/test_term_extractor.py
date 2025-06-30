import pytest
from unittest.mock import patch, MagicMock
from internal.term_extractor.term_extractor import MBartTermExtractor

sample_text = "Предложение до. Алгоритмы и структуры данных являются основой компьютерных наук. Предложение после."

@pytest.fixture
def extractor():
    return MBartTermExtractor()

@patch('internal.term_extractor.term_extractor.Client')
def test_extract_terms(mock_client, extractor):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "алгоритмы, структуры данных, компьютерные науки"
    mock_response.choices = [mock_choice]
    instance = mock_client.return_value
    instance.chat.completions.create.return_value = mock_response

    terms = extractor.extract_terms(sample_text)
    assert terms == ["Алгоритмы", "структуры данных", "компьютерные науки."]

def test_lemmatize_text(extractor):
    lemmatized = extractor.lemmatize_text("Алгоритмы и структуры данных")
    expected = "алгоритм и структура данные"
    assert lemmatized == expected


@patch('internal.term_extractor.term_extractor.Doc', autospec=True)
def test_extract_sentences_with_terms(mock_doc, extractor):
    mock_sent1 = MagicMock()
    mock_sent1.text = "Алгоритмы и структуры данных являются основой компьютерных наук."
    mock_sent2 = MagicMock()
    mock_sent2.text = "Они используются в различных областях."
    mock_doc_instance = mock_doc.return_value
    mock_doc_instance.sents = [mock_sent1, mock_sent2]

    with patch.object(extractor, 'lemmatize_text', side_effect=lambda x: x.lower()):
        terms = ["алгоритмы", "структуры данных"]
        sentences = extractor.extract_sentences_with_terms(sample_text, terms)
        expected = {
            "алгоритмы": [
                "Алгоритмы и структуры данных являются основой компьютерных наук.",
                "Они используются в различных областях."
            ],
            "структуры данных": [
                "Алгоритмы и структуры данных являются основой компьютерных наук.",
                "Они используются в различных областях."
            ]
        }
        assert sentences == expected


sample_text_1 = (
    "В ходе исследования были изучены биоритмы человека и их влияние на "
    "эффективность работы в условиях гиперконкурентного рынка. "
    "Также рассмотрены аспекты нейролингвистического программирования и "
    "их применение в маркетинговых стратегиях."
)


@patch('internal.term_extractor.term_extractor.Client')
def test_extract_dif_terms(mock_client, extractor):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_response.choices = [mock_choice]
    instance = mock_client.return_value
    instance.chat.completions.create.return_value = mock_response

    terms = extractor.extract_terms(sample_text_1)
    expected_terms = [
        "",
    ]
    assert terms == expected_terms