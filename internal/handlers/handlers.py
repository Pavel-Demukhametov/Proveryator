# internal/handlers/handlers.py

from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Union
from internal.test_generator import TestGenerator
from internal.schemas import (
    GeneralTestCreationRequest,
    ByThemesTestCreationRequest,
    Theme,
    Question,
    TestCreationResponse
)
import shutil
import logging
from internal.utils.text_processing import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import os
import random
from internal.utils.answer_generation import generate_incorrect_answers


logger = logging.getLogger(__name__)


test_generator = TestGenerator()

async def handle_lecture_upload(file: UploadFile, materials: Optional[str]):
    logger.info(f"File: {file.filename if file else None}")

    if not file:
        raise HTTPException(
            status_code=400, detail="Файл лекции обязателен."
        )

    if not materials:
        raise HTTPException(
            status_code=400, detail="Материалы лекции обязательны."
        )

    try:
        os.makedirs("uploaded_files", exist_ok=True)
        file_path = os.path.join("uploaded_files", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Файл {file.filename} успешно сохранен.")

        # Определение типа файла и извлечение текста
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.pdf':
            extracted_text = extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            extracted_text = extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            extracted_text = extract_text_from_txt(file_path)
        else:
            # Для остальных форматов предполагается, что файл содержит текст
            with open(file_path, "r", encoding='utf-8') as f:
                extracted_text = f.read()

        logger.info(f"Текст успешно извлечён из файла {file.filename}.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении или обработке файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении или обработке файла: {e}")

    try:
        # Если материал предоставлен как дополнительный ввод, можно объединить с извлечённым текстом
        combined_materials = f"{materials}\n\n{extracted_text}"
        keywords = test_generator.extract_keywords(combined_materials)
        logger.info(f"Извлечено {len(keywords)} ключевых слов.")
    except Exception as e:
        logger.error(f"Ошибка при извлечении ключевых слов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при извлечении ключевых слов: {e}")

    results = {}
    segments = []
    for keyword in keywords:
        keyword_combined = ' '.join(word.get_word() for word in keyword.words)
        keyword_lemmatized = test_generator.tokenize_lemmatize(keyword_combined)
        sentences_with_keyword = test_generator.extract_sentences_with_keyword(combined_materials, keyword_lemmatized)

        if sentences_with_keyword:
            results[keyword.normalized] = sentences_with_keyword
            segments.append({
                "keyword": keyword.normalized,
                "sentences": sentences_with_keyword 
            })

    response_data = {
        "message": "Лекция успешно обработана!",
        "file_name": file.filename,
        "materials": combined_materials,
        "results": results,
        "segments": segments
    }

    return JSONResponse(content=response_data, status_code=200)


async def handle_test_creation(test_request: Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]):
    """
    Обработчик для создания теста.
    Генерирует вопросы на основе предоставленных материалов лекции.
    """
    try:
        if isinstance(test_request, GeneralTestCreationRequest):
            # Для метода 'general', генерируем вопросы на основе всего материала
            generated_questions = test_generator.process_text(test_request.lectureMaterials)
            logger.info(f"Сгенерировано {len(generated_questions)} вопросов для теста.")

            # Фильтруем вопросы по типу
            multiple_choice_questions = [q for q in generated_questions if q["type"] == "mc"]
            open_answer_questions = [q for q in generated_questions if q["type"] == "open"]

            # Выбираем необходимое количество вопросов
            selected_mc = multiple_choice_questions[:test_request.multipleChoiceCount]
            selected_open = open_answer_questions[:test_request.openAnswerCount]

            test_questions = selected_mc + selected_open

            # Структурирование вопросов для ответа с добавлением неверных вариантов для "mc" вопросов
            structured_questions = []
            for q in test_questions:
                if q["type"] == "mc":
                    # Используем функцию из модуля utils для генерации вариантов ответов
                    options = generate_incorrect_answers(correct_answer=q["answer"], num_incorrect=3)
                    # Добавляем в объект вопроса
                    structured_question = Question(
                        type=q["type"],
                        question=q["question"],
                        answer=q["answer"],      # Правильный ответ
                        options=options,          # Варианты ответов
                        sentence=q["sentence"]
                    )
                else:
                    # Для открытых вопросов
                    structured_question = Question(
                        type=q["type"],
                        question=q["question"],
                        answer=q["answer"],
                        sentence=q["sentence"]
                    )
                structured_questions.append(structured_question)

            # Структурирование тем для ответа
            structured_themes = [
                Theme(
                    keyword=theme.keyword,
                    sentences=theme.sentences,
                    multipleChoiceCount=test_request.multipleChoiceCount,
                    openAnswerCount=test_request.openAnswerCount
                ) for theme in test_request.themes
            ]

            response_data = TestCreationResponse(
                message="Тест успешно создан.",
                method=test_request.method,
                title=test_request.title,
                lectureMaterials=test_request.lectureMaterials,
                questions=[q.dict() for q in structured_questions],
                themes=structured_themes
            )

            return JSONResponse(content=response_data.dict(), status_code=200)

        elif isinstance(test_request, ByThemesTestCreationRequest):
            # Для метода 'byThemes', генерируем вопросы по каждой теме
            themes_data = [
                {
                    "keyword": theme.keyword,
                    "sentences": theme.sentences,
                    "multipleChoiceCount": theme.multipleChoiceCount,
                    "openAnswerCount": theme.openAnswerCount
                } for theme in test_request.themes
            ]

            generated_questions = test_generator.process_text_by_theme(themes_data)
            logger.info(f"Сгенерировано {len(generated_questions)} вопросов для теста по темам.")

            # Структурирование вопросов для ответа с добавлением неверных вариантов для "mc" вопросов
            structured_questions = []
            for q in generated_questions:
                if q["type"] == "mc":
                    # Используем функцию из модуля utils для генерации вариантов ответов
                    options = generate_incorrect_answers(correct_answer=q["answer"], num_incorrect=3)
                    # Добавляем в объект вопроса
                    structured_question = Question(
                        type=q["type"],
                        question=q["question"],
                        answer=q["answer"],      # Правильный ответ
                        options=options,          # Варианты ответов
                        sentence=q["sentence"]
                    )
                else:
                    # Для открытых вопросов
                    structured_question = Question(
                        type=q["type"],
                        question=q["question"],
                        answer=q["answer"],
                        sentence=q["sentence"]
                    )
                structured_questions.append(structured_question)

            # Структурирование тем для ответа
            structured_themes = [
                Theme(
                    keyword=theme.keyword,
                    sentences=theme.sentences,
                    multipleChoiceCount=theme.multipleChoiceCount,
                    openAnswerCount=theme.openAnswerCount
                ) for theme in test_request.themes
            ]

            response_data = TestCreationResponse(
                message="Тест по выбранным темам успешно создан.",
                method=test_request.method,
                title=test_request.title,
                lectureMaterials=test_request.lectureMaterials,
                questions=[q.dict() for q in structured_questions],
                themes=structured_themes
            )

            return JSONResponse(content=response_data.dict(), status_code=200)

        else:
            raise HTTPException(status_code=400, detail="Неверный формат запроса.")

    except Exception as e:
        logger.error(f"Ошибка при создании теста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при создании теста: {e}")