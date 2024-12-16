# internal/handlers/handlers.py

from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Union
from internal.keywords_extractor import KeywordsExtractor
from internal.schemas import (
    GeneralTestCreationRequest,
    ByThemesTestCreationRequest,
    Theme,
    Question,
    TestCreationResponse
)
from internal.qa_generator import QAGenerator  # Используем QAGenerator
from internal.utils.text_converter import AudioTranscription  # Импортируем AudioTranscription
import shutil
import logging
from internal.utils.text_converter import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import os
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

# Инициализируем генераторы
qa_generator = QAGenerator()
audio_transcriber = AudioTranscription()
keywords_extractor = KeywordsExtractor()

logger = logging.getLogger(__name__)

# Создайте глобальный ThreadPoolExecutor для CPU-емких задач
executor = ThreadPoolExecutor(max_workers=8)  # Настройте количество потоков в зависимости от нагрузки

async def handle_lecture_upload(file: Optional[UploadFile], materials: Optional[str]) -> JSONResponse:
    if not file and not materials:
        raise HTTPException(
            status_code=400, detail="Пожалуйста, загрузите файл лекции или введите материалы лекции."
        )

    extracted_text = ""

    try:
        if file:
            os.makedirs("uploaded_files", exist_ok=True)
            file_path = os.path.join("uploaded_files", file.filename)
            async with aiofiles.open(file_path, "wb") as buffer:
                content = await file.read()
                await buffer.write(content)

            file_extension = os.path.splitext(file.filename)[1].lower()

            if file_extension == '.pdf':
                extracted_text = await asyncio.get_event_loop().run_in_executor(
                    executor, extract_text_from_pdf, file_path
                )
            elif file_extension == '.docx':
                extracted_text = await asyncio.get_event_loop().run_in_executor(
                    executor, extract_text_from_docx, file_path
                )
            elif file_extension == '.txt':
                extracted_text = await asyncio.get_event_loop().run_in_executor(
                    executor, extract_text_from_txt, file_path
                )
            elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                audio_extracted_path = os.path.join("uploaded_files", f"{os.path.splitext(file.filename)[0]}.wav")
                await asyncio.get_event_loop().run_in_executor(
                    executor, audio_transcriber.extract_audio_from_video, file_path, audio_extracted_path
                )
                transcription = await asyncio.get_event_loop().run_in_executor(
                    executor, audio_transcriber.transcribe, audio_extracted_path
                )
                extracted_text = transcription
                logger.info(f"Транскрибированный текст из видео файла {file.filename} успешно получен.")
            elif file_extension in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                wav_path = os.path.join("uploaded_files", f"{os.path.splitext(file.filename)[0]}.wav")
                await asyncio.get_event_loop().run_in_executor(
                    executor, audio_transcriber.convert_audio_to_wav, file_path, wav_path
                )
                transcription = await asyncio.get_event_loop().run_in_executor(
                    executor, audio_transcriber.transcribe, wav_path
                )
                extracted_text = transcription
                logger.info(f"Транскрибированный текст из аудио файла {file.filename} успешно получен.")
            else:
                try:
                    async with aiofiles.open(file_path, "r", encoding='utf-8') as f:
                        extracted_text = await f.read()
                    logger.info(f"Текст успешно извлечён из файла {file.filename}.")
                except Exception as e:
                    logger.error(f"Ошибка при чтении текстового файла {file.filename}: {e}")
                    raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат файла или ошибка при чтении файла {file.filename}.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении или обработке файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении или обработке файла: {e}")

    try:
        if materials:
            combined_materials = materials
            if extracted_text:
                combined_materials += f"\n\n{extracted_text}"
        else:
            combined_materials = extracted_text

        logger.info(f"Объединённые материалы лекции:\n{combined_materials}")

        # Асинхронное извлечение ключевых слов
        keywords = await asyncio.get_event_loop().run_in_executor(
            executor, keywords_extractor.extract_keywords, combined_materials
        )
        logger.info(f"Извлечено {len(keywords)} ключевых слов.")
    except Exception as e:
        logger.error(f"Ошибка при извлечении ключевых слов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при извлечении ключевых слов: {e}")

    results = {}
    segments = []
    for keyword in keywords:
        keyword_combined = keyword  # Используем keyword напрямую, так как это строка
        keyword_lemmatized = keywords_extractor.tokenize_lemmatize(keyword_combined)
        sentences_with_keyword = await asyncio.get_event_loop().run_in_executor(
            executor, keywords_extractor.extract_sentences_with_keyword, combined_materials, keyword_lemmatized
        )

        if sentences_with_keyword:
            results[keyword] = sentences_with_keyword  # Используем keyword напрямую
            segments.append({
                "keyword": keyword,
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


async def handle_test_creation(test_request: Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]) -> JSONResponse:
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
            existing_embeddings = []  # Список векторных представлений уже сгенерированных вопросов

            MAX_ATTEMPTS = 3  # Максимальное количество попыток перегенерации похожих вопросов
            SIMILARITY_THRESHOLD = 0.9  # Порог схожести


            for theme in themes:
                remaining_mc = total_mc - sum(1 for q in generated_questions if q.get("type") == "mc")
                remaining_oa = total_oa - sum(1 for q in generated_questions if q.get("type") == "open")

                if remaining_mc <= 0 and remaining_oa <= 0:
                    break

                mc_to_generate = min(remaining_mc, max(1, int(total_mc / len(themes)))) if total_mc > 0 else 0
                oa_to_generate = min(remaining_oa, max(1, int(total_oa / len(themes)))) if total_oa > 0 else 0

                # Генерация MC вопросов с проверкой на схожесть
                theme_sentences = theme.sentences
                if mc_to_generate > 0:
                    for _ in range(mc_to_generate):
                        attempts = 0
                        while attempts < MAX_ATTEMPTS:
                            print(len(theme_sentences))
                            theme_text = ' '.join(theme_sentences)
                            try:
                                mc_question = await asyncio.get_event_loop().run_in_executor(
                                    executor, qa_generator.generate_qa_pairs,
                                    theme_text, 1, 'mc'
                                )
                                question = mc_question[0]
                                question_embedding = qa_generator.get_sentence_embedding(question["Вопрос"])
                                
                                if question_embedding is None:
                                    logger.warning("Не удалось получить векторное представление вопроса. Перегенерируем.")
                                    attempts += 1
                                    continue

                                if not qa_generator.is_similar(existing_embeddings, question_embedding, SIMILARITY_THRESHOLD):
                                    # Уникальный вопрос
                                    generated_questions.append(question)
                                    existing_embeddings.append(question_embedding)
                                    logger.debug(f"Сгенерирован уникальный MC вопрос из темы '{theme.keyword}': {question['Вопрос']}")
                                    break  # Выход из цикла попыток
                                else:
                                    logger.debug(f"Найдена похожая MC вопрос из темы '{theme.keyword}': {question['Вопрос']}. Перегенерируем.")
                                    attempts += 1

                                # Обрезаем 33% оставшихся предложений
                                # theme_sentences = theme_sentences[int(len(theme_sentences) * 2 / 3):]
                                theme_sentences = theme_sentences[2:]
                                if not theme_sentences:  # Если оставшихся предложений нет, выходим
                                    break
                            except Exception as e:
                                logger.error(f"Ошибка при генерации MC вопроса из темы '{theme.keyword}': {e}")
                                attempts += 1
                        else:
                            logger.warning(f"Не удалось сгенерировать уникальный MC вопрос после {MAX_ATTEMPTS} попыток из темы '{theme.keyword}'.")

                # Генерация Open вопросов с проверкой на схожесть
                if oa_to_generate > 0:
                    for _ in range(oa_to_generate):
                        attempts = 0
                        while attempts < MAX_ATTEMPTS:
                            try:
                                print(len(theme_sentences))
                                theme_text = ' '.join(theme_sentences)
                                oa_question = await asyncio.get_event_loop().run_in_executor(
                                    executor, qa_generator.generate_qa_pairs,
                                    theme_text, 1, 'open'
                                )
                                question = oa_question[0]
                                question_embedding = qa_generator.get_sentence_embedding(question["Вопрос"])
                                
                                if question_embedding is None:
                                    logger.warning("Не удалось получить векторное представление вопроса. Перегенерируем.")
                                    attempts += 1
                                    continue

                                if not qa_generator.is_similar(existing_embeddings, question_embedding, SIMILARITY_THRESHOLD):
                                    # Уникальный вопрос
                                    generated_questions.append(question)
                                    existing_embeddings.append(question_embedding)
                                    logger.debug(f"Сгенерирован уникальный Open вопрос из темы '{theme.keyword}': {question['Вопрос']}")
                                    break  # Выход из цикла попыток
                                else:
                                    logger.debug(f"Найдена похожая Open вопрос из темы '{theme.keyword}': {question['Вопрос']}. Перегенерируем.")
                                    attempts += 1

                                # Обрезаем 33% оставшихся предложений
                                # theme_sentences = theme_sentences[int(len(theme_sentences) * 2 / 3):]
                                theme_sentences = theme_sentences[2:]
                                if not theme_sentences:  # Если оставшихся предложений нет, выходим
                                    break
                            except Exception as e:
                                logger.error(f"Ошибка при генерации Open вопроса из темы '{theme.keyword}': {e}")
                                attempts += 1
                        else:
                            logger.warning(f"Не удалось сгенерировать уникальный Open вопрос после {MAX_ATTEMPTS} попыток из темы '{theme.keyword}'.")


            # Проверка, удалось ли сгенерировать требуемое количество вопросов
            actual_mc = sum(1 for q in generated_questions if q.get("type") == "mc")
            actual_oa = sum(1 for q in generated_questions if q.get("type") == "open")

            if actual_mc < total_mc or actual_oa < total_oa:
                logger.warning("Не удалось сгенерировать требуемое количество уникальных вопросов из предоставленных тем.")
                raise HTTPException(
                    status_code=400,
                    detail=f"Не удалось сгенерировать требуемое количество уникальных вопросов. Требуется: {total_mc} MC и {total_oa} Open, сгенерировано: {actual_mc} MC и {actual_oa} Open."
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
            # Аналогичные изменения для генерации по темам отдельно
            themes_data = [
                {
                    "keyword": theme.keyword,
                    "sentences": theme.sentences,
                    "multipleChoiceCount": theme.multipleChoiceCount,
                    "openAnswerCount": theme.openAnswerCount
                } for theme in test_request.themes
            ]

            qa_pairs = []
            existing_embeddings = []  # Список векторных представлений уже сгенерированных вопросов

            MAX_ATTEMPTS = 3
            SIMILARITY_THRESHOLD = 0.9
            theme_sentences = theme["sentences"]
            for theme in themes_data:
                theme_text = ' '.join(theme["sentences"])


                # Генерация MC вопросов
                for _ in range(theme["multipleChoiceCount"]):
                    attempts = 0
                    while attempts < MAX_ATTEMPTS:
                        theme_sentences = theme_sentences[3:]
                        print(len(theme_sentences))
                        # Например, взять первые 3 предложения
                        theme_text = ' '.join(theme_sentences)
                        try:
                            theme_qas_mc = await asyncio.get_event_loop().run_in_executor(
                                executor, qa_generator.generate_qa_pairs,
                                theme_text, 1, 'mc'
                            )
                            question = theme_qas_mc[0]
                            question_embedding = qa_generator.get_sentence_embedding(question["Вопрос"])
                            
                            if question_embedding is None:
                                logger.warning("Не удалось получить векторное представление вопроса. Перегенерируем.")
                                attempts += 1
                                continue

                            if not qa_generator.is_similar(existing_embeddings, question_embedding, SIMILARITY_THRESHOLD):
                                # Уникальный вопрос
                                qa_pairs.append(question)
                                existing_embeddings.append(question_embedding)
                                logger.debug(f"Сгенерирован уникальный MC вопрос из темы '{theme['keyword']}': {question['Вопрос']}")
                                break
                            else:
                                logger.debug(f"Найдена похожая MC вопрос из темы '{theme['keyword']}': {question['Вопрос']}. Перегенерируем.")
                                attempts += 1
                        except Exception as e:
                            logger.error(f"Ошибка при генерации MC вопроса из темы '{theme['keyword']}': {e}")
                            attempts += 1
                    else:
                        logger.warning(f"Не удалось сгенерировать уникальный MC вопрос после {MAX_ATTEMPTS} попыток из темы '{theme['keyword']}'.")

                # Генерация Open вопросов
                for _ in range(theme["openAnswerCount"]):
                    attempts = 0
                    while attempts < MAX_ATTEMPTS:
                        theme_sentences = theme_sentences[3:]
                        print(len(theme_sentences))
                        # Например, взять первые 3 предложения
                        theme_text = ' '.join(theme_sentences)
                        try:
                            theme_qas_oa = await asyncio.get_event_loop().run_in_executor(
                                executor, qa_generator.generate_qa_pairs,
                                theme_text, 1, 'open'
                            )
                            question = theme_qas_oa[0]
                            question_embedding = qa_generator.get_sentence_embedding(question["Вопрос"])
                            
                            if question_embedding is None:
                                logger.warning("Не удалось получить векторное представление вопроса. Перегенерируем.")
                                attempts += 1
                                continue

                            if not qa_generator.is_similar(existing_embeddings, question_embedding, SIMILARITY_THRESHOLD):
                                # Уникальный вопрос
                                qa_pairs.append(question)
                                existing_embeddings.append(question_embedding)
                                logger.debug(f"Сгенерирован уникальный Open вопрос из темы '{theme['keyword']}': {question['Вопрос']}")
                                break
                            else:
                                logger.debug(f"Найдена похожая Open вопрос из темы '{theme['keyword']}': {question['Вопрос']}. Перегенерируем.")
                                attempts += 1
                        except Exception as e:
                            logger.error(f"Ошибка при генерации Open вопроса из темы '{theme['keyword']}': {e}")
                            attempts += 1
                    else:
                        logger.warning(f"Не удалось сгенерировать уникальный Open вопрос после {MAX_ATTEMPTS} попыток из темы '{theme['keyword']}'.")

            # Проверка, удалось ли сгенерировать требуемое количество вопросов
            actual_mc = sum(1 for q in qa_pairs if q.get("type") == "mc")
            actual_oa = sum(1 for q in qa_pairs if q.get("type") == "open")

            if actual_mc < sum(theme["multipleChoiceCount"] for theme in themes_data) or \
               actual_oa < sum(theme["openAnswerCount"] for theme in themes_data):
                logger.warning("Не удалось сгенерировать требуемое количество уникальных вопросов из предоставленных тем.")
                raise HTTPException(
                    status_code=400,
                    detail=f"Не удалось сгенерировать требуемое количество уникальных вопросов. Требуется: {sum(theme['multipleChoiceCount'] for theme in themes_data)} MC и {sum(theme['openAnswerCount'] for theme in themes_data)} Open, сгенерировано: {actual_mc} MC и {actual_oa} Open."
                )

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
                questions=[q.dict() for q in structured_questions],
                themes=structured_themes
            )

            return JSONResponse(content=response_data.dict(), status_code=200)

        else:
            raise HTTPException(status_code=400, detail="Неверный формат запроса.")

    except Exception as e:
        logger.error(f"Ошибка при создании теста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при создании теста: {e}")