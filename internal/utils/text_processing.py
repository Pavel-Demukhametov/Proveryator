# internal/handlers/text_processing.py

import logging
import os
from typing import Optional

from docx import Document
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Извлекает текст из PDF файла.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста из PDF: {e}")
        raise e


def extract_text_from_docx(file_path: str) -> str:
    """
    Извлекает текст из DOCX файла.
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста из DOCX: {e}")
        raise e


def extract_text_from_txt(file_path: str) -> str:
    """
    Извлекает текст из TXT файла.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста из TXT: {e}")
        raise e
