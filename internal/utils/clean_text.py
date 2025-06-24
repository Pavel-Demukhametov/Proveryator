import re
from cleantext import clean


def clean_sent(input_text: str) -> str:
    """
    Очищает текст от кодов, оглавлений, мусора, строк с многоточиями и лишних символов.
    """

    cleaned_text = input_text

    cleaned_text = re.sub(r"```[\s\S]*?```", "", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"`[^`]*?`", "", cleaned_text)
    cleaned_text = re.sub(r'^.*\.{5,}.*$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\d+\s+.+?\.{2,}\s*\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\s*(Глава|Раздел)\s*\d+(\.\d+)*:?.*$', '', cleaned_text, flags=re.MULTILINE)

    cleaned_text = re.sub(r"(Источники|Список литературы)[\s\S]*", "", cleaned_text, flags=re.IGNORECASE)

    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)

    cleaned_text = re.sub(r'[©©️]+', '', cleaned_text)
    cleaned_text = re.sub(r'[^\S\r\n]{2,}', ' ', cleaned_text) 
    cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text).strip()

    return cleaned_text