# internal/handlers/handlers.py

from fastapi import UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from typing import Optional, List, Union
from internal.term_extractor.term_extractor import MBartTermExtractor
from internal.schemas import (
    GeneralTestCreationRequest,
    ByThemesTestCreationRequest,
    Theme,
    Question,
    TestCreationResponse
)
from internal.qa_generator.qa_generator import QAGenerator
from internal.utils.embedding import EmbeddingManager
import shutil
import logging
from internal.utils.text_converter import AudioTranscription, extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import os
import re
import random
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from internal.text_ranger.text_ranger import rank_sentences_with_index, split_into_sentences, remove_duplicate_sentences
qa_generator = QAGenerator()
embedding_manager = EmbeddingManager()
audio_transcriber = AudioTranscription()
mbart_extractor = MBartTermExtractor()

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=8)


def clean_text(text: str) -> str:
    cleaned_text = re.sub(r'[\r\n\t]', ' ', text)
    cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    return cleaned_text.strip()


async def handle_lecture_upload(file: Optional[UploadFile], materials: Optional[str]) -> JSONResponse:
    if not file and not materials:
        raise HTTPException(
            status_code=400, detail="Пожалуйста, загрузите файл лекции или введите материалы лекции."
        )

    extracted_text = ""

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
        elif file_extension in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
            wav_path = os.path.join("uploaded_files", f"{os.path.splitext(file.filename)[0]}.wav")
            await asyncio.get_event_loop().run_in_executor(
                executor, audio_transcriber.convert_audio_to_wav, file_path, wav_path
            )
            transcription = await asyncio.get_event_loop().run_in_executor(
                executor, audio_transcriber.transcribe, wav_path
            )
            extracted_text = transcription
        else:
            try:
                async with aiofiles.open(file_path, "r", encoding='utf-8') as f:
                    extracted_text = await f.read()
            except Exception as e:
                logger.error(f"Ошибка при чтении текстового файла {file.filename}: {e}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Неподдерживаемый формат файла или ошибка при чтении файла {file.filename}."
                )

    # Очищаем извлечённый текст от управляющих символов
    extracted_text = clean_text(extracted_text)

    if materials:
        # Очищаем входные материалы
        materials = clean_text(materials)
        combined_materials = materials
        if extracted_text:
            combined_materials += f" {extracted_text}"
    else:
        combined_materials = extracted_text

    # Проверяем, что материалов достаточно для извлечения терминов
    if len(combined_materials.split()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Ошибка: материалов слишком мало для извлечения терминов (менее 50 слов)."
        )

    # Извлекаем термины с помощью mbart_extractor
    terms = await asyncio.get_event_loop().run_in_executor(
        executor, mbart_extractor.extract_terms, combined_materials
    )
    
    # Если термины не удалось извлечь, возвращаем ошибку
    if not terms:
        raise HTTPException(
            status_code=400,
            detail="Ошибка: не удалось извлечь термины из предоставленных материалов."
        )

    term_sentences = await asyncio.get_event_loop().run_in_executor(
        executor, mbart_extractor.extract_sentences_with_terms, combined_materials, terms
    )

    segments = []
    for term, sentences in term_sentences.items():
        if sentences:
            segments.append({
                "keyword": term,
                "sentences": sentences
            })

    response_data = {
        "message": "Лекция успешно обработана!",
        "file_name": file.filename if file else None,
        "materials": combined_materials,
        "results": term_sentences,
        "segments": segments
    }

    return JSONResponse(content=response_data, status_code=200)

async def handle_test_creation(
    test_request: Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]
) -> JSONResponse:
    """
    Обработчик для создания теста.
    
    При методе по темам (ByThemesTestCreationRequest) глобальное ранжирование предложений 
    производится по объединённым ключевым словам из всех тем, но при генерации вопросов для 
    каждой темы используется только то предложение, в котором содержится ключевое слово этой темы.
    """
    qa_pairs = []
    source_text = test_request.lectureMaterials
    sentences = split_into_sentences(source_text)
    sentences = remove_duplicate_sentences(sentences)
    global_keyword = []
    if hasattr(test_request, "themes") and test_request.themes:
        for theme in test_request.themes:
            if hasattr(theme, "keyword") and theme.keyword:
                global_keyword.append(theme.keyword)
            else:
                global_keyword.append(theme.keyword)
    global_keyword = list(set(global_keyword))
    ranked_with_idx = rank_sentences_with_index(sentences, global_keyword)

    structured_themes = []
    structured_questions = []

    def is_duplicate(question_text: str) -> bool:
        return any(q.get("Вопрос") == question_text for q in qa_pairs)
    MAX_RETRIES = 3

    if isinstance(test_request, GeneralTestCreationRequest):
        total_mc = test_request.multipleChoiceCount
        total_oa = test_request.openAnswerCount
        mc_generated = 0
        oa_generated = 0

        for original_idx, sent in ranked_with_idx:
            if mc_generated >= total_mc and oa_generated >= total_oa:
                break
            context_text = build_context_window(sentences, original_idx)
            if mc_generated < total_mc:
                retries = 0
                while retries < MAX_RETRIES:
                    qa_mc = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        qa_generator.generate_qa,
                        context_text,
                        global_keyword,
                        False
                    )
                    if qa_mc and "Вопрос" in qa_mc:
                        if is_duplicate(qa_mc["Вопрос"]):
                            new_idx = original_idx + 1 if original_idx + 1 < len(sentences) else original_idx - 1
                            context_text = build_context_window(sentences, new_idx)
                            retries += 1
                            continue
                        else:
                            qa_pairs.append({ "type": "mc", **qa_mc })
                            mc_generated += 1
                    break

            context_text = build_context_window(sentences, original_idx)
            if oa_generated < total_oa:
                retries = 0
                while retries < MAX_RETRIES:
                    qa_oa = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        qa_generator.generate_qa,
                        context_text,
                        global_keyword,
                        True
                    )
                    if qa_oa and "Вопрос" in qa_oa:
                        if is_duplicate(qa_oa["Вопрос"]):
                            new_idx = original_idx + 1 if original_idx + 1 < len(sentences) else original_idx - 1
                            context_text = build_context_window(sentences, new_idx)
                            retries += 1
                            continue
                        else:
                            qa_pairs.append({ "type": "open", **qa_oa })
                            oa_generated += 1
                    break

        if mc_generated < total_mc or oa_generated < total_oa:
            raise HTTPException(
                status_code=400,
                detail=f"Требуется: {total_mc} MC и {total_oa} Open, сгенерировано: {mc_generated} MC и {oa_generated} Open."
            )
        structured_themes.append(Theme(
            keyword=global_keyword,
            lectureMaterials=source_text,
            multipleChoiceCount=total_mc,
            openAnswerCount=total_oa
        ))

    elif isinstance(test_request, ByThemesTestCreationRequest):
        for theme in test_request.themes:
            theme_keyword = theme.keyword

            mc_needed = theme.multipleChoiceCount
            oa_needed = theme.openAnswerCount
            mc_generated = 0
            oa_generated = 0

            for original_idx, sent in ranked_with_idx:
                if not any(keyword in sent for keyword in theme_keyword.split()):
                    continue

                if mc_generated >= mc_needed and oa_generated >= oa_needed:
                    break

                context_text = build_context_window(sentences, original_idx)

                if mc_generated < mc_needed:
                    retries = 0
                    while retries < MAX_RETRIES:
                        qa_mc = await asyncio.get_event_loop().run_in_executor(
                            executor,
                            qa_generator.generate_qa,
                            context_text,
                            theme_keyword,
                            False
                        )
                        if qa_mc and "Вопрос" in qa_mc:
                            if is_duplicate(qa_mc["Вопрос"]):
                                new_idx = original_idx + 1 if original_idx + 1 < len(sentences) else original_idx - 1
                                context_text = build_context_window(sentences, new_idx)
                                retries += 1
                                continue
                            else:
                                qa_pairs.append({
                                    "type": "mc",
                                    "theme": theme.keyword,
                                    "sentence": sent,
                                    **qa_mc
                                })
                                mc_generated += 1
                        break

                context_text = build_context_window(sentences, original_idx)
                if oa_generated < oa_needed:
                    retries = 0
                    while retries < MAX_RETRIES:
                        qa_oa = await asyncio.get_event_loop().run_in_executor(
                            executor,
                            qa_generator.generate_qa,
                            context_text,
                            theme_keyword,
                            True
                        )
                        if qa_oa and "Вопрос" in qa_oa:
                            if is_duplicate(qa_oa["Вопрос"]):
                                new_idx = original_idx + 1 if original_idx + 1 < len(sentences) else original_idx - 1
                                context_text = build_context_window(sentences, new_idx)
                                retries += 1
                                continue
                            else:
                                qa_pairs.append({
                                    "type": "open",
                                    "theme": theme.keyword,
                                    "sentence": sent,
                                    **qa_oa
                                })
                                oa_generated += 1
                        break

            structured_themes.append(Theme(
                keyword=theme_keyword,
                lectureMaterials=source_text,
                multipleChoiceCount=mc_needed,
                openAnswerCount=oa_needed
            ))
    else:
        raise HTTPException(status_code=400, detail="Неверный формат запроса.")
    if not qa_pairs:
        raise HTTPException(
            status_code=400,
            detail="Не удалось сгенерировать ни одного вопроса."
        )
    for pair in qa_pairs:
        if pair["type"] == "mc":
            options = pair["Варианты"]
            correct_answer = options[pair["Правильный_ответ"]]
            structured_questions.append(Question(
                type="mc",
                question=pair["Вопрос"],
                answer=correct_answer,
                options=options,
                sentence=pair.get("sentence", pair["Вопрос"]),
                theme=pair.get("theme")
            ))
        elif pair["type"] == "open":
            structured_questions.append(Question(
                type="open",
                question=pair["Вопрос"],
                answer=pair["Правильный_ответ"],
                sentence=pair.get("sentence", pair["Вопрос"]),
                theme=pair.get("theme")
            ))

    response_data = TestCreationResponse(
        message="Тест успешно создан.",
        method=test_request.method,
        title=test_request.title,
        lectureMaterials=source_text,
        questions=structured_questions,
        themes=structured_themes
    )
    return JSONResponse(content=response_data.dict(), status_code=200)
async def handle_websocket_transcription(websocket: WebSocket):
    await audio_transcriber.transcribe_audio_stream(websocket)

def build_context_window(sentences: list[str], center_index: int, min_words: int = 80) -> str:
    """
    Формирует контекстное окно из списка предложений.
    Если центральное предложение содержит меньше min_words слов, то добавляются соседние предложения (сначала слева, затем справа),
    пока суммарное число слов не станет не меньше min_words.
    
    :param sentences: Список предложений полного текста.
    :param center_index: Индекс центрального предложения.
    :param min_words: Минимальное требуемое количество слов в контексте (по умолчанию 20).
    :return: Строка, содержащая объединённые предложения.
    """
    def word_count(text: str) -> int:
        return len(text.split())
    
    # Начинаем с центрального предложения
    context = [sentences[center_index]]
    total_words = word_count(sentences[center_index])
    
    left = center_index - 1
    right = center_index + 1
    
    # Пока не достигнуто нужное количество слов и есть предложения для добавления
    while total_words < min_words and (left >= 0 or right < len(sentences)):
        # Добавляем сначала предыдущее предложение, если оно есть
        if left >= 0:
            context.insert(0, sentences[left])
            total_words += word_count(sentences[left])
            left -= 1
            if total_words >= min_words:
                break
        # Затем добавляем следующее предложение, если оно есть
        if right < len(sentences):
            context.append(sentences[right])
            total_words += word_count(sentences[right])
            right += 1

    return " ".join(context)