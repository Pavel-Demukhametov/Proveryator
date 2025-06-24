# internal/handlers/handlers.py

from fastapi import UploadFile, HTTPException, WebSocket
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional, List, Union
from internal.term_extractor.term_extractor import TermExtractor
from internal.schemas import (
    GeneralTestCreationRequest,
    ByThemesTestCreationRequest,
    Theme,
    Question,
    TestCreationResponse
)
from internal.utils.clean_text import clean_sent
from internal.qa_generator.qa_generator import QAGenerator
from internal.utils.embedding import EmbeddingManager
import shutil
import logging
from internal.utils.text_converter import AudioTranscription, extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx
import os, uuid
import re
import random
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from internal.entity_linker.entity_linker import EntityLinker
from internal.text_ranger.text_ranger import rank_sentences_with_index, split_into_sentences, remove_duplicate_sentences
qa_generator = QAGenerator()
embedding_manager = EmbeddingManager()
audio_transcriber = AudioTranscription()
term_extractor = TermExtractor()
entity_linker = EntityLinker() 

logger = logging.getLogger(__name__)
router = APIRouter()
executor = ThreadPoolExecutor(max_workers=8)

TestRequest = Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]

@router.post("/lectures/", status_code=202)
def clean_text(text: str) -> str:
    cleaned_text = re.sub(r'[\r\n\t]', ' ', text)
    cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    return cleaned_text.strip()


progress_map = {}
result_map = {}

def clean_text(text: str) -> str:
    cleaned_text = re.sub(r'[\r\n\t]', ' ', text)
    cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    return cleaned_text.strip()


@router.post("/upload/", status_code=202)
async def handle_lecture_upload(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = None,
    materials: Optional[str] = None
):
    if not files and not materials:
        raise HTTPException(status_code=400, detail="Пожалуйста, загрузите файлы лекции или введите материалы лекции.")

    os.makedirs("uploaded_files", exist_ok=True)
    file_paths = []
    if files:
        for file in files:
            file_path = os.path.join("uploaded_files", file.filename)
            async with aiofiles.open(file_path, "wb") as buffer:
                content = await file.read()
                await buffer.write(content)
            file_paths.append(file_path)

    task_id = str(uuid.uuid4())
    progress_map[task_id] = 0.0
    background_tasks.add_task(executor.submit, run_transcription_task, task_id, file_paths, materials)
    return {"task_id": task_id}

@router.get("/progress/{task_id}")
def get_progress(task_id: str):
    progress = progress_map.get(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return {"progress": progress}

@router.get("/result/{task_id}")
def get_result(task_id: str):
    result = result_map.get(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Результат ещё не готов")
    return result


def run_transcription_task(task_id: str, file_paths: List[str], materials: Optional[str]):
    try:
        extracted_texts = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            def progress_cb(p):
                if p < 1.0:
                    progress_map[task_id] = round(p, 4)
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                audio_path = os.path.splitext(file_path)[0] + ".wav"
                audio_transcriber.extract_audio_from_video(file_path, audio_path)
                extracted_text = audio_transcriber.transcribe(audio_path, progress_callback=progress_cb)
            elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                wav_path = os.path.splitext(file_path)[0] + ".wav"
                audio_transcriber.convert_audio_to_wav(file_path, wav_path)
                extracted_text = audio_transcriber.transcribe(wav_path, progress_callback=progress_cb)
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            elif ext == '.pdf':
                extracted_text = extract_text_from_pdf(file_path)
            elif ext == '.docx':
                extracted_text = extract_text_from_docx(file_path)
            elif ext == '.pptx':
                extracted_text = extract_text_from_pptx(file_path)
            else:
                raise Exception(f"Неподдерживаемый формат файла: {ext}")
            extracted_texts.append(extracted_text)

        combined_materials = (materials or "") + "\n" + "\n".join(extracted_texts)
        combined_materials = clean_sent(combined_materials)

        raw_segments = term_extractor.extract_terms(combined_materials)
        valid_segments = term_extractor.validate_terms(combined_materials, raw_segments, entity_linker)

        segments = []
        for item in valid_segments:
            if isinstance(item, dict):
                keyword = item.get('keyword', '')
                sentences = item.get('sentences') or []
            else:
                keyword = str(item)
                sentences = []
            segments.append({
                'keyword': keyword,
                'sentences': sentences
            })

        # Сохранение результата
        result_map[task_id] = {
            "segments": segments,
            "materials": combined_materials
        }
        progress_map[task_id] = 1.0
    except Exception as e:
        logger.error(f"Error (task_id={task_id}): {e}")
        result_map[task_id] = {"error": str(e)}
        progress_map[task_id] = 1.0


async def _run_generation(task_id: str, test_request: Union[GeneralTestCreationRequest, ByThemesTestCreationRequest], progress_map, result_map):
    qa_pairs = []
    source_text = test_request.lectureMaterials
    sentences = split_into_sentences(source_text)
    sentences = remove_duplicate_sentences(sentences)
    global_keyword = []
    
    if hasattr(test_request, "themes") and test_request.themes:
        for theme in test_request.themes:
            if hasattr(theme, "keyword") and theme.keyword:
                global_keyword.append(theme.keyword)
        global_keyword = list(set(global_keyword))
    
    ranked_with_idx = rank_sentences_with_index(sentences, global_keyword)
    structured_themes = []
    structured_questions = []
    
    def is_duplicate(question_text: str) -> bool:
        return any(q.get("Вопрос") == question_text for q in qa_pairs)
    
    MAX_RETRIES = 6

    if isinstance(test_request, GeneralTestCreationRequest):
        mc_total = test_request.multipleChoiceCount
        oa_total = test_request.openAnswerCount
    else:
        mc_total = sum(t.multipleChoiceCount for t in test_request.themes)
        oa_total = sum(t.openAnswerCount for t in test_request.themes)
    
    total = mc_total + oa_total
    done = 0
    mc_generated = 0
    oa_generated = 0
    
    if isinstance(test_request, GeneralTestCreationRequest):
        for original_idx, sent in ranked_with_idx:
            if mc_generated >= mc_total and oa_generated >= oa_total:
                break
            
            context_text = build_context_window(sentences, original_idx)
            
            if mc_generated < mc_total:
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
                            qa_pairs.append({"type": "mc", **qa_mc})
                            mc_generated += 1
                            done += 1
                            progress_map[task_id] = done / total if total > 0 else 1.0
                            break
                    else:
                        logger.warning(f"Failed to generate MC question for sentence at index {original_idx}, skipping.")
                        retries += 1
            
            context_text = build_context_window(sentences, original_idx)
            if oa_generated < oa_total:
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
                            qa_pairs.append({"type": "open", **qa_oa})
                            oa_generated += 1
                            done += 1
                            progress_map[task_id] = done / total if total > 0 else 1.0
                            break
                    else:
                        logger.warning(f"Failed to generate Open question for sentence at index {original_idx}, skipping.")
                        retries += 1
    
        if mc_generated < mc_total or oa_generated < oa_total:
            logger.warning(f"Required: {mc_total} MC and {oa_total} Open, generated: {mc_generated} MC and {oa_generated} Open.")
        
        structured_themes.append(Theme(
            keyword=global_keyword,
            lectureMaterials=source_text,
            multipleChoiceCount=mc_total,
            openAnswerCount=oa_total
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
                                done += 1
                                progress_map[task_id] = done / total if total > 0 else 1.0
                                break
                        else:
                            logger.warning(f"Failed to generate MC question for sentence at index {original_idx} for theme '{theme_keyword}', skipping.")
                            retries += 1
                
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
                                done += 1
                                progress_map[task_id] = done / total if total > 0 else 1.0
                                break
                        else:
                            logger.warning(f"Failed to generate Open question for sentence at index {original_idx} for theme '{theme_keyword}', skipping.")
                            retries += 1
            
            structured_themes.append(Theme(
                keyword=theme_keyword,
                lectureMaterials=source_text,
                multipleChoiceCount=mc_needed,
                openAnswerCount=oa_needed
            ))
    
    else:
        logger.warning("Invalid request format, proceeding with empty results.")
        result_map[task_id] = TestCreationResponse(
            message="Invalid request format, no questions generated.",
            method=test_request.method,
            title=test_request.title,
            lectureMaterials=source_text,
            questions=[],
            themes=[]
        )
        progress_map[task_id] = 1.0
        return
    if not qa_pairs:
        logger.warning("No questions generated for the test.")
        result_map[task_id] = TestCreationResponse(
            message="No questions generated.",
            method=test_request.method,
            title=test_request.title,
            lectureMaterials=source_text,
            questions=[],
            themes=structured_themes
        )
        progress_map[task_id] = 1.0
        return
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
            options = pair.get("Варианты", [])
            idx = pair["Правильный_ответ"]
            if options and isinstance(idx, int) and 0 <= idx < len(options):
                correct_answer = options[idx]
            else:
                correct_answer = str(pair.get("Правильный_ответ", ""))
            structured_questions.append(Question(
                type="open",
                question=pair["Вопрос"],
                answer=correct_answer,
                sentence=pair.get("sentence", pair["Вопрос"]),
                theme=pair.get("theme")
            ))
    response_data = TestCreationResponse(
        message=f"Test created successfully. Generated: {len([q for q in qa_pairs if q['type'] == 'mc'])} MC and {len([q for q in qa_pairs if q['type'] == 'open'])} Open.",
        method=test_request.method,
        title=test_request.title,
        lectureMaterials=source_text,
        questions=structured_questions,
        themes=structured_themes
    )
    result_map[task_id] = response_data
    progress_map[task_id] = 1.0


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
    print("хуй1")
    if isinstance(test_request, GeneralTestCreationRequest):
        total_mc = test_request.multipleChoiceCount
        total_oa = test_request.openAnswerCount
        mc_generated = 0
        oa_generated = 0
        print("хуй2")
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
            print("хуй ", context_text)
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
        print("хуй3")
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

def build_context_window(sentences: list[str], center_index: int, min_words: int = 50) -> str:

    def word_count(text: str) -> int:
        return len(text.split())

    context = [sentences[center_index]]
    total_words = word_count(sentences[center_index])
    
    left = center_index - 1
    right = center_index + 1
    
    while total_words < min_words and (left >= 0 or right < len(sentences)):
        if left >= 0:
            context.insert(0, sentences[left])
            total_words += word_count(sentences[left])
            left -= 1
            if total_words >= min_words:
                break
        if right < len(sentences):
            context.append(sentences[right])
            total_words += word_count(sentences[right])
            right += 1

    return " ".join(context)