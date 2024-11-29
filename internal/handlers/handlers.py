from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from internal.models import draft
from transformers import AutoTokenizer, T5ForConditionalGeneration
from rutermextract import TermExtractor
from functools import partial
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
import torch

class Theme(BaseModel):
    keyword: str
    sentences: List[str]
    questionCount: Optional[int] = None

class TestCreationRequest(BaseModel):
    method: str
    title: str
    totalQuestions: Optional[str] = None
    themes: Optional[List[Theme]] = None
    lectureMaterials: Optional[List[str]] = None

async def handle_lecture_upload(method: str, url: str, file: UploadFile, materials: str):
    print(f"url: {url}")
    print(f"materials: {materials}")
    print(f"file: {file.filename if file else None}")

    if not materials:
        raise HTTPException(
            status_code=400, detail="Материалы лекции обязательны."
        )

    if method == 'YandexDisk' and not url:
        raise HTTPException(
            status_code=400, detail="Для метода YandexDisk требуется ссылка."
        )

    if method == 'Device' and not file:
        raise HTTPException(
            status_code=400, detail="Для метода Device требуется файл."
        )

    stop_words = set(stopwords.words('russian'))
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_vocab = MorphVocab()
    morph_tagger = NewsMorphTagger(emb)

    doc = Doc(materials)
    doc.segment(segmenter)

    keywords = draft.extract_keywords(materials)

    results = {}

    for keyword in keywords:
        keyword_combined = ' '.join(word.get_word() for word in keyword.words)
        keyword_lemmatized = draft.tokenize_lemmatize(keyword_combined, segmenter, morph_tagger, morph_vocab)
        sentences_with_keyword = draft.extract_sentences_with_keyword(doc, keyword_lemmatized, segmenter, morph_tagger, morph_vocab)

        if sentences_with_keyword:
            results[keyword.normalized] = sentences_with_keyword

    response_data = {
        "message": "Лекция успешно обработана!",
        "method": method,
        "url": url if method == 'YandexDisk' else None,
        "file_name": file.filename if file else None,
        "materials": materials,
        "results": results,
        "segments": [
            {
                "keyword": keyword.normalized,
                "sentences": sentences_with_keyword  # Сами предложения, связанные с ключевым словом
            }
            for keyword in keywords
        ]
    }

    return JSONResponse(content=response_data, status_code=200)
    
async def handle_test_creation(
    method: str, 
    title: str, 
    totalQuestions: Optional[str], 
    themes: Optional[List[Theme]], 
    lectureMaterials: Optional[str]
):
    if not title:
        raise HTTPException(status_code=400, detail="Название теста обязательно.")

    if method not in ["general", "byThemes"]:
        raise HTTPException(status_code=400, detail="Неверный метод создания теста.")

    # Проверка для метода "general"
    if method == "general":
        if totalQuestions is None or totalQuestions <= 0:
            raise HTTPException(status_code=400, detail="Укажите корректное количество вопросов.")

    # Проверка для метода "byThemes"
    elif method == "byThemes":
        if not themes or not isinstance(themes, list):
            raise HTTPException(status_code=400, detail="Темы обязательны при методе byThemes.")
        
        if len(themes) == 0:
            raise HTTPException(status_code=400, detail="Должна быть выбрана хотя бы одна тема.")
        print('круто')
    #     for theme in themes:
    #             # Проверка для количества вопросов с одним правильным ответом
    #             if theme.questionCountSingle is None or theme.questionCountSingle <= 0:
    #                 raise HTTPException(status_code=400, detail=f"Укажите корректное количество вопросов с одним правильным ответом для темы '{theme.title}'.")

    #             # Проверка для количества вопросов с открытым ответом
    #             if theme.questionCountOpen is None or theme.questionCountOpen <= 0:
    #                 raise HTTPException(status_code=400, detail=f"Укажите корректное количество вопросов с открытым ответом для темы '{theme.title}'.")

    # # Печать информации о запросе
    # print(f"Метод: {method}")
    # print(f"Название: {title}")

    # if method == "general":
    #     print(f"Количество вопросов: {totalQuestions}")
    # elif method == "byThemes":
    #     print("Вопросы по темам:")
    #     for theme in themes:
    #         if theme.isIncluded:
    #             print(f"  Тема: {theme.title}, Количество вопросов с одним правильным ответом: {theme.questionCountSingle}, Количество вопросов с открытым ответом: {theme.questionCountOpen}")
    
    # print(f"Материалы лекции: {lectureMaterials}")

    # # Обработка текстовых материалов лекции (если есть)
    # draft.process_text(lectureMaterials)

    # # Формирование ответа
    # response_data = {
    #     "message": "Тест успешно создан.",
    #     "testId": 1,  # Тут будет логика для создания теста и генерации уникального ID
    #     "method": method,
    #     "title": title,
    #     "totalQuestions": totalQuestions,
    #     "themes": [theme.dict() for theme in themes] if themes else None,
    #     "lectureMaterials": lectureMaterials,
    # }

    # Возвращаем успешный ответ
    return JSONResponse(content="some", status_code=200)