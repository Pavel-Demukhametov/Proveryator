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
from internal.utils.audio_transcription import AudioTranscription  # Импортируем AudioTranscription
import shutil
import logging
from internal.utils.text_processing import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import os
import random
from internal.utils.answer_generation import generate_incorrect_answers
qa_generator = QAGenerator()


logger = logging.getLogger(__name__)

audio_transcriber = AudioTranscription()  # Инициализируем транскрибера
test_generator = TestGenerator()

async def handle_lecture_upload(file: Optional[UploadFile], materials: Optional[str]):
    logger.info(f"File: {file.filename if file else None}")

    # Валидация: должно быть хотя бы одно из полей
    if not file and not materials:
        raise HTTPException(
            status_code=400, detail="Пожалуйста, загрузите файл лекции или введите материалы лекции."
        )

    extracted_text = ""

    try:
        if file:
            # Сохранение файла на диск
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
            elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                # Обработка видео файлов
                audio_extracted_path = os.path.join("uploaded_files", f"{os.path.splitext(file.filename)[0]}.wav")
                audio_transcriber.extract_audio_from_video(file_path, audio_extracted_path)
                transcription = audio_transcriber.transcribe(audio_extracted_path)
                extracted_text = transcription
                logger.info(f"Транскрибированный текст из видео файла {file.filename} успешно получен.")
            elif file_extension in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                # Обработка аудио файлов
                wav_path = os.path.join("uploaded_files", f"{os.path.splitext(file.filename)[0]}.wav")
                audio_transcriber.convert_audio_to_wav(file_path, wav_path)
                transcription = audio_transcriber.transcribe(wav_path)
                extracted_text = transcription
                logger.info(f"Транскрибированный текст из аудио файла {file.filename} успешно получен.")
            else:
                # Для остальных форматов предполагается, что файл содержит текст
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        extracted_text = f.read()
                    logger.info(f"Текст успешно извлечён из файла {file.filename}.")
                except Exception as e:
                    logger.error(f"Ошибка при чтении текстового файла {file.filename}: {e}")
                    raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат файла или ошибка при чтении файла {file.filename}.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении или обработке файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении или обработке файла: {e}")

    try:
        # Если материал предоставлен как дополнительный ввод, можно объединить с извлечённым текстом
        if materials:
            combined_materials = materials
            if extracted_text:
                combined_materials += f"\n\n{extracted_text}"
        else:
            combined_materials = extracted_text  # Если только файл или только материалы

        logger.info(f"Объединённые материалы лекции:\n{combined_materials}")

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
        "file_name": file.filename if file else None,
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
            total_mc = test_request.multipleChoiceCount
            total_oa = test_request.openAnswerCount

            if total_mc <= 0 and total_oa <= 0:
                raise HTTPException(status_code=400, detail="Необходимо указать хотя бы одно количество вопросов: с одним правильным ответом или с открытым ответом.")

            themes = test_request.themes.copy()
            random.shuffle(themes)

            generated_questions = []

            for theme in themes:
                remaining_mc = total_mc - sum(1 for q in generated_questions if q.get("type") == "mc")
                remaining_oa = total_oa - sum(1 for q in generated_questions if q.get("type") == "open")

                if remaining_mc <= 0 and remaining_oa <= 0:
                    break

                mc_to_generate = min(remaining_mc, max(1, int(total_mc / len(themes)))) if total_mc > 0 else 0
                oa_to_generate = min(remaining_oa, max(1, int(total_oa / len(themes)))) if total_oa > 0 else 0

                # Генерация MC вопросов
                if mc_to_generate > 0:
                    try:
                        mc_questions = qa_generator.generate_qa_pairs(
                            text=' '.join(theme.sentences),
                            num_questions=mc_to_generate,
                            question_type='mc'
                        )
                        generated_questions.extend(mc_questions)
                        logger.debug(f"Сгенерировано {len(mc_questions)} MC вопросов из темы '{theme.keyword}'.")
                    except Exception as e:
                        logger.error(f"Ошибка при генерации MC вопросов из темы '{theme.keyword}': {e}")

                # Генерация Open вопросов
                if oa_to_generate > 0:
                    try:
                        oa_questions = qa_generator.generate_qa_pairs(
                            text=' '.join(theme.sentences),
                            num_questions=oa_to_generate,
                            question_type='open'
                        )
                        generated_questions.extend(oa_questions)
                        logger.debug(f"Сгенерировано {len(oa_questions)} Open вопросов из темы '{theme.keyword}'.")
                    except Exception as e:
                        logger.error(f"Ошибка при генерации Open вопросов из темы '{theme.keyword}': {e}")

            # Логирование всех сгенерированных вопросов
            logger.debug(f"Итого сгенерировано вопросов: {generated_questions}")

            actual_mc = sum(1 for q in generated_questions if q.get("type") == "mc")
            actual_oa = sum(1 for q in generated_questions if q.get("type") == "open")

            if actual_mc < total_mc or actual_oa < total_oa:
                logger.warning("Не удалось сгенерировать требуемое количество вопросов из предоставленных тем.")
                raise HTTPException(
                    status_code=400,
                    detail=f"Не удалось сгенерировать требуемое количество вопросов. Требуется: {total_mc} MC и {total_oa} Open, сгенерировано: {actual_mc} MC и {actual_oa} Open."
                )

            # Структурирование вопросов для ответа
            structured_questions = []
            for pair in generated_questions:
                if pair["type"] == "mc":
                    # Генерация вариантов ответов
                    distractors = pair.get("Дистракторы") or []
                    options = [opt for opt in distractors if opt]
                    # Добавляем правильный ответ и перемешиваем варианты
                    all_options = options + [pair["Ответ"]]
                    random.shuffle(all_options)
                    structured_question = Question(
                        type="mc",
                        question=pair["Вопрос"],
                        answer=pair["Ответ"],
                        options=all_options,
                        sentence=pair.get("sentence", pair["Вопрос"])
                    )
                elif pair["type"] == "open":
                    # Для открытых вопросов
                    structured_question = Question(
                        type="open",
                        question=pair["Вопрос"],
                        answer=pair["Ответ"],
                        sentence=pair.get("sentence", pair["Вопрос"])
                    )
                else:
                    logger.error(f"Неизвестный тип вопроса: {pair['type']}")
                    continue  # Пропустить неизвестные типы вопросов
                structured_questions.append(structured_question)

            response_data = TestCreationResponse(
                message="Тест успешно создан.",
                method=test_request.method,
                title=test_request.title,
                lectureMaterials=test_request.lectureMaterials,
                questions=[q.dict() for q in structured_questions],
                themes=[Theme(
                    keyword=theme.keyword,
                    sentences=theme.sentences,
                    multipleChoiceCount=0,  # Устанавливаем в 0 для общего метода
                    openAnswerCount=0        # Устанавливаем в 0 для общего метода
                ).dict() for theme in test_request.themes]
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
                # Генерируем вопросы с одним правильным ответом
                if theme["multipleChoiceCount"] > 0:
                    try:
                        theme_qas_mc = qa_generator.generate_qa_pairs(
                            theme_text,
                            num_questions=theme["multipleChoiceCount"],
                            question_type='mc'
                        )
                        qa_pairs.extend(theme_qas_mc)
                        logger.debug(f"Сгенерировано {len(theme_qas_mc)} MC вопросов из темы '{theme['keyword']}'.")
                    except Exception as e:
                        logger.error(f"Ошибка при генерации MC вопросов из темы '{theme['keyword']}': {e}")

                # Генерируем открытые вопросы
                if theme["openAnswerCount"] > 0:
                    try:
                        theme_qas_oa = qa_generator.generate_qa_pairs(
                            theme_text,
                            num_questions=theme["openAnswerCount"],
                            question_type='open'
                        )
                        qa_pairs.extend(theme_qas_oa)
                        logger.debug(f"Сгенерировано {len(theme_qas_oa)} Open вопросов из темы '{theme['keyword']}'.")
                    except Exception as e:
                        logger.error(f"Ошибка при генерации Open вопросов из темы '{theme['keyword']}': {e}")

            # Структурирование вопросов
            structured_questions = []
            for pair in qa_pairs:
                if pair["type"] == "mc":
                    # Добавляем правильный ответ к дистракторам и перемешиваем
                    distractors = pair.get("Дистракторы") or []
                    options = [opt for opt in distractors if opt]
                    all_options = options + [pair["Ответ"]]
                    random.shuffle(all_options)
                    structured_question = Question(
                        type="mc",
                        question=pair["Вопрос"],
                        answer=pair["Ответ"],
                        options=all_options,
                        sentence=pair.get("sentence", pair["Вопрос"])
                    )
                elif pair["type"] == "open":
                    structured_question = Question(
                        type="open",
                        question=pair["Вопрос"],
                        answer=pair["Ответ"],
                        sentence=pair.get("sentence", pair["Вопрос"])
                    )
                else:
                    logger.error(f"Неизвестный тип вопроса: {pair['type']}")
                    continue  # Пропустить неизвестные типы вопросов
                structured_questions.append(structured_question)

            # Структурирование тем для ответа
            structured_themes = [
                Theme(
                    keyword=theme["keyword"],
                    sentences=theme["sentences"],
                    multipleChoiceCount=theme["multipleChoiceCount"],
                    openAnswerCount=theme["openAnswerCount"]
                ).dict() for theme in themes_data
            ]

            response_data = TestCreationResponse(
                message="Тест по выбранным темам успешно создан.",
                method=test_request.method,
                title=test_request.title,
                lectureMaterials=test_request.lectureMaterials,
                questions=structured_questions,  # Убрали .dict()
                themes=structured_themes  # Убрали .dict()
            )

            return JSONResponse(content=response_data.dict(), status_code=200)

        else:
            raise HTTPException(status_code=400, detail="Неверный формат запроса.")

    except Exception as e:
        logger.error(f"Ошибка при создании теста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при создании теста: {e}")