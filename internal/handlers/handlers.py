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
from internal.qa_generator import QAGenerator  # Используем QAGenerator
import shutil
import logging
from internal.utils.text_processing import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import os
import random
from internal.utils.answer_generation import generate_incorrect_answers
qa_generator = QAGenerator()


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
    Генерирует вопросы на основе предоставленных материалов лекции или по темам.
    """
    try:
        if isinstance(test_request, GeneralTestCreationRequest):
            # Для общего метода генерации теста используем весь текст
            combined_materials = test_request.lectureMaterials

            # Генерируем QA пары
            generated_questions = qa_generator.generate_qa_pairs(combined_materials, num_questions=test_request.numQuestions)

            # Структурирование вопросов для ответа
            structured_questions = []
            for pair in generated_questions:
                if pair["Дистракторы"]:
                    # Генерируем варианты ответов (дистракторы)
                    options = pair["Дистракторы"]
                    # Убираем None значения из дистракторов
                    options = [opt for opt in options if opt]
                    # Добавляем правильный ответ и перемешиваем варианты
                    all_options = options + [pair["Ответ"]]
                    random.shuffle(all_options)
                    structured_question = Question(
                        type="mc",
                        question=pair["Вопрос"],
                        answer=pair["Ответ"],
                        options=all_options,
                        sentence=pair["Вопрос"]  # Можно заменить на соответствующее предложение, если доступно
                    )
                else:
                    # Для открытых вопросов
                    structured_question = Question(
                        type="open",
                        question=pair["Вопрос"],
                        answer=pair["Ответ"],
                        sentence=pair["Вопрос"]  # Можно заменить на соответствующее предложение, если доступно
                    )
                structured_questions.append(structured_question)

            response_data = TestCreationResponse(
                message="Тест успешно создан.",
                method=test_request.method,
                title=test_request.title,
                lectureMaterials=test_request.lectureMaterials,
                questions=[q.dict() for q in structured_questions],
                themes=[]  # Для общего метода темы могут быть не нужны
            )

            return JSONResponse(content=response_data.dict(), status_code=200)

        elif isinstance(test_request, ByThemesTestCreationRequest):
            themes_data = [
                {
                    "keyword": theme.keyword,
                    "sentences": theme.sentences,
                    "multipleChoiceCount": theme.multipleChoiceCount,
                    "openAnswerCount": theme.openAnswerCount
                } for theme in test_request.themes
            ]

            # Генерация вопросов по каждой теме отдельно
            qa_pairs = []
            for theme in themes_data:
                theme_text = ' '.join(theme["sentences"])
                num_questions = theme["multipleChoiceCount"] + theme["openAnswerCount"]
                theme_qas = qa_generator.generate_qa_pairs(theme_text, num_questions=num_questions)
                for pair in theme_qas:
                    if pair["Дистракторы"]:
                        pair_type = "mc"
                    else:
                        pair_type = "open"
                    qa_pairs.append({
                        "type": pair_type,
                        "question": pair["Вопрос"],
                        "answer": pair["Ответ"],
                        "sentence": theme_text,
                        "options": [opt for opt in pair["Дистракторы"] if opt] if pair["Дистракторы"] else []
                    })

            # Структурирование вопросов
            structured_questions = []
            for pair in qa_pairs:
                if pair["type"] == "mc":
                    # Добавляем правильный ответ к дистракторам и перемешиваем
                    all_options = pair["options"] + [pair["answer"]]
                    random.shuffle(all_options)
                    structured_question = Question(
                        type="mc",
                        question=pair["question"],
                        answer=pair["answer"],
                        options=all_options,
                        sentence=pair["sentence"]
                    )
                else:
                    structured_question = Question(
                        type="open",
                        question=pair["question"],
                        answer=pair["answer"],
                        sentence=pair["sentence"]
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
                themes=[t.dict() for t in structured_themes]
            )

            return JSONResponse(content=response_data.dict(), status_code=200)

        else:
            raise HTTPException(status_code=400, detail="Неверный формат запроса.")

    except Exception as e:
        logger.error(f"Ошибка при создании теста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при создании теста: {e}")