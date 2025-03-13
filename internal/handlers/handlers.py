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
from internal.qa_generator.qa_generator import ChatGPTQAGenerator
from internal.utils.embedding import EmbeddingManager
import shutil
import logging
from internal.utils.text_converter import AudioTranscription, extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import os
import random
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
import aiofiles

qa_generator = ChatGPTQAGenerator()
embedding_manager = EmbeddingManager()
audio_transcriber = AudioTranscription()
mbart_extractor = MBartTermExtractor()

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=8)


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
                raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат файла или ошибка при чтении файла {file.filename}.")



    if materials:
        combined_materials = materials
        if extracted_text:
            combined_materials += f"\n\n{extracted_text}"
    else:
        combined_materials = extracted_text
    terms = await asyncio.get_event_loop().run_in_executor(
        executor, mbart_extractor.extract_terms, combined_materials
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


async def handle_test_creation(test_request: Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]) -> JSONResponse:
    """
    Обработчик для создания теста.
    Генерирует вопросы на основе предоставленных тем или общего количества.
    """
    qa_pairs = []
    if isinstance(test_request, GeneralTestCreationRequest):
        themes_data = [
            {
                "keyword": theme.keyword,
                "sentences": theme.sentences
            } for theme in test_request.themes
        ]

        total_mc = test_request.multipleChoiceCount
        total_oa = test_request.openAnswerCount
        themes_count = len(themes_data)

        if themes_count == 0:
            raise HTTPException(status_code=400, detail="Список тем пуст.")

        mc_per_theme = total_mc // themes_count
        oa_per_theme = total_oa // themes_count
        extra_mc = total_mc % themes_count
        extra_oa = total_oa % themes_count
        for i, theme in enumerate(themes_data):
            full_text = ' '.join(theme["sentences"])
            mc_count = mc_per_theme + (1 if i < extra_mc else 0)
            oa_count = oa_per_theme + (1 if i < extra_oa else 0)

            for _ in range(mc_count):
                theme_qas_mc = await asyncio.get_event_loop().run_in_executor(
                    executor, qa_generator.generate_qa,
                    full_text, theme["keyword"], False
                )
                if theme_qas_mc and "Вопрос" in theme_qas_mc:
                    qa_pairs.append({"type": "mc", **theme_qas_mc})

            for _ in range(oa_count):
                theme_qas_oa = await asyncio.get_event_loop().run_in_executor(
                    executor, qa_generator.generate_qa,
                    full_text, theme["keyword"], True
                )
                if theme_qas_oa and "Вопрос" in theme_qas_oa:
                    qa_pairs.append({"type": "open", **theme_qas_oa})

    elif isinstance(test_request, ByThemesTestCreationRequest):
        themes_data = [
            {
                "keyword": theme.keyword,
                "sentences": theme.sentences,
                "multipleChoiceCount": theme.multipleChoiceCount,
                "openAnswerCount": theme.openAnswerCount
            } for theme in test_request.themes
        ]

        for theme in themes_data:
            full_text = ' '.join(theme["sentences"])
            for _ in range(theme["multipleChoiceCount"]):
                theme_qas_mc = await asyncio.get_event_loop().run_in_executor(
                    executor, qa_generator.generate_qa,
                    full_text, theme["keyword"], False
                )
                if theme_qas_mc and "Вопрос" in theme_qas_mc:
                    qa_pairs.append({"type": "mc", **theme_qas_mc})
            for _ in range(theme["openAnswerCount"]):
                theme_qas_oa = await asyncio.get_event_loop().run_in_executor(
                    executor, qa_generator.generate_qa,
                    full_text, theme["keyword"], True
                )
                if theme_qas_oa and "Вопрос" in theme_qas_oa:
                    qa_pairs.append({"type": "open", **theme_qas_oa})
    else:
        raise HTTPException(status_code=400, detail="Неверный формат запроса.")

    actual_mc = sum(1 for q in qa_pairs if q["type"] == "mc")
    actual_oa = sum(1 for q in qa_pairs if q["type"] == "open")
    required_mc = test_request.multipleChoiceCount if isinstance(test_request, GeneralTestCreationRequest) else sum(theme["multipleChoiceCount"] for theme in themes_data)
    required_oa = test_request.openAnswerCount if isinstance(test_request, GeneralTestCreationRequest) else sum(theme["openAnswerCount"] for theme in themes_data)

    if actual_mc < required_mc or actual_oa < required_oa:
        raise HTTPException(
            status_code=400,
            detail=f"Требуется: {required_mc} MC и {required_oa} Open, сгенерировано: {actual_mc} MC и {actual_oa} Open."
        )

    structured_questions = []
    for pair in qa_pairs:
        if pair["type"] == "mc":
            options = pair["Варианты"]
            correct_answer = options[pair["Правильный_ответ"]]
            structured_questions.append(Question(
                type="mc",
                question=pair["Вопрос"],
                answer=correct_answer,
                options=options,
                sentence=pair.get("sentence", pair["Вопрос"])
            ))
        elif pair["type"] == "open":
            structured_questions.append(Question(
                type="open",
                question=pair["Вопрос"],
                answer=pair["Правильный_ответ"],
                sentence=pair.get("sentence", pair["Вопрос"])
            ))

    structured_themes = [
        Theme(
            keyword=theme["keyword"],
            sentences=theme["sentences"],
            multipleChoiceCount=theme.get("multipleChoiceCount", 0),
            openAnswerCount=theme.get("openAnswerCount", 0)
        ).dict() for theme in themes_data
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

async def handle_websocket_transcription(websocket: WebSocket):
    await audio_transcriber.transcribe_audio_stream(websocket)