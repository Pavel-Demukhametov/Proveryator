# main.py
from fastapi import (
    FastAPI, Form, File, UploadFile, HTTPException, Request, Depends,
    status, BackgroundTasks, WebSocket
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import ValidationError

from internal.handlers.handlers import (
    handle_lecture_upload,
    handle_test_creation,
    handle_websocket_transcription,
)
from internal.auth.registration import handle_registration
from internal.auth.authentication import handle_login
from internal.schemas import (
    TestCreationRequest,
    ByThemesTheme,
    GeneralTestCreationRequest,
    ByThemesTestCreationRequest,
    UserCreate,
    UserResponse,
    UserLogin,
    Token,
    TokenData,
    QuestionTypeResponse
)
from internal.term_extractor.term_extractor import MBartTermExtractor
from internal.database import get_db
from internal.utils.user import get_current_user
from internal.utils.gift_generation import convert_to_gift
from internal.repositories.auth_repository import SECRET_KEY, ALGORITHM
from internal.repositories.user_repository import get_user_by_email
from internal.repositories.test_repository import get_all_test_types

from typing import List, Optional, Union
import asyncpg
import aiofiles
import asyncio
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import urllib
from urllib.parse import quote
import jwt
from jwt import PyJWTError
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from fastapi import APIRouter

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

minio_client = Minio(
    endpoint="localhost:9000",       # или ваш хост/порт
    access_key="admin",              # замените на свой
    secret_key="password",           # замените на свой
    secure=False                     # True, если используете HTTPS
)

BUCKET_NAME = "proveryator"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login/")
keywords_extractor = MBartTermExtractor()
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if hasattr(handler, 'setEncoding'):
    handler.setEncoding('utf-8')

logger.addHandler(handler)

executor = ThreadPoolExecutor(max_workers=8)


async def get_current_user_dependency(token: str = Depends(oauth2_scheme), conn: asyncpg.Connection = Depends(get_db)) -> UserResponse:
    """
    Получает текущего пользователя на основе JWT токена.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный токен.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except PyJWTError:
        raise credentials_exception
    user = await get_user_by_email(conn, token_data.email)
    if user is None:
        raise credentials_exception
    return UserResponse(id=user["id"], username=user["username"], email=user["email"])

@app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket_transcription(websocket)

@app.get("/ping")
async def ping():
    return {"message": "pong"}

FILE_PATH = "D:\\Program Files\\Lecture_test_front\\fast\\uploaded_files\\v2.docx"

@app.get("/download-example")
async def download_file():
    if not os.path.exists(FILE_PATH):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(FILE_PATH, media_type="application/octet-stream", filename="v2.docx")

@app.post("/api/register/", status_code=201)
async def register_user(request: Request, conn: asyncpg.Connection = Depends(get_db)):
    """
    Регистрация нового пользователя.
    """

    try:
        user_data = await request.json()
        user = UserCreate(**user_data)
    except ValidationError as ve:
        logger.error(f"Ошибка валидации данных: {ve}")
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        logger.error(f"Ошибка при получении данных из запроса: {e}")
        raise HTTPException(status_code=400, detail="Некорректный формат запроса")

    return await handle_registration(user, conn)
    
@app.post("/api/login/", response_model=Token)
async def login(user: UserLogin, conn: asyncpg.Connection = Depends(get_db)):
    """
    Логин пользователя и получение JWT токена.
    """
    return await handle_login(user, conn)

@app.post("/api/upload/")
async def lecture_upload(
    file: Optional[UploadFile] = File(None),
    materials: Optional[str] = Form(None)
):
    return await handle_lecture_upload(file, materials)

@app.post("/api/tests/download/")
async def download_gift_file(request: Request):
    """
    Создаёт файл GIFT на лету из предоставленных данных и отправляет его клиенту для загрузки.
    """
    try:
        data = await request.json()
        test_title = data.get("title", "default_test").strip()
        if not test_title:
            raise HTTPException(status_code=400, detail="Не указано название теста.")
        sanitized_test_title = "".join(c for c in test_title if c.isalnum() or c in (" ", "_", "-")).rstrip()
        file_name = f"{sanitized_test_title}.gift"
        encoded_file_name = quote(file_name.encode('utf-8'))
        questions = data.get("questions", [])
        lecture_materials = data.get("lectureMaterials", [])

        gift_content = await asyncio.get_event_loop().run_in_executor(
            executor, convert_to_gift, questions
        )
        gift_file = io.BytesIO(gift_content.encode("utf-8"))
        encoded_filename = urllib.parse.quote(file_name)

        content_disposition = f"attachment; filename=\"{encoded_filename}\""

        return StreamingResponse(
            gift_file,
            media_type="application/octet-stream",
            headers={"Content-Disposition": content_disposition}
        )

    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Ошибка при создании файла GIFT: {e}")
        raise HTTPException(status_code=500, detail="Произошла ошибка при создании файла GIFT.")

@app.post("/api/tests/create/")
async def test_creation(
    request: Request,
    conn: asyncpg.Connection = Depends(get_db),
):
    """
    Создание теста. Доступно без авторизации.
    """
    try:
        data = await request.json()
        method = data.get("method")
        if method not in ["general", "byThemes"]:
            raise HTTPException(status_code=400, detail="Неверный метод создания теста.")

        if method == "general":
            try:
                test_request = GeneralTestCreationRequest(**data)
            except ValidationError as ve:
                logger.error(f"Валидационная ошибка: {ve}")
                raise HTTPException(status_code=400, detail=ve.errors())
        elif method == "byThemes":
            results = {}
            segments = []
            combined_materials = data.get('lectureMaterials', '')
            themes_data = data.get('themes', [])
            for theme in themes_data:
                keyword = theme['keyword']
                multiple_choice_count = theme.get('multipleChoiceCount', 0)
                open_answer_count = theme.get('openAnswerCount', 0)
                sentences_with_keyword = keywords_extractor.extract_sentences_with_term(combined_materials, keyword)
                if sentences_with_keyword:
                    results[keyword] = sentences_with_keyword
                    segments.append({
                        "keyword": keyword,
                        "sentences": sentences_with_keyword,
                        "multipleChoiceCount": multiple_choice_count,
                        "openAnswerCount": open_answer_count
                    })
         
            try:
                test_request_data = {
                    "method": "byThemes", 
                    "title": data.get('title', ''),
                    "lectureMaterials": combined_materials,
                    "themes": segments
                }
                test_request = ByThemesTestCreationRequest(**test_request_data)
                
            except ValidationError as ve:
                logger.error(f"Validation error: {ve}")
                raise HTTPException(status_code=400, detail=ve.errors())
        else:
            raise HTTPException(status_code=400, detail="Неверный метод создания теста.")
        response = await asyncio.get_event_loop().run_in_executor(
            executor, handle_test_creation_sync, test_request
        )
        return response

    except json.JSONDecodeError:
        logger.error("Некорректный формат JSON.")
        raise HTTPException(status_code=400, detail="Некорректный формат JSON.")
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_test_creation_sync(test_request: Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]):
    """
    Синхронный обработчик для создания теста.
    Выполняется в отдельном потоке, чтобы не блокировать event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(handle_test_creation(test_request))
        return response
    finally:
        loop.close()

@app.post("/api/tests/save/")
async def test_save(
    request: Request,
    current_user = Depends(get_current_user_dependency)
):
    """Сохранение теста в MinIO (без user_id в ответе)."""
    try:
        data = await request.json()
        test_name = data.get("test_name", "default_test").strip()
        questions = data.get("questions", [])

        if not test_name:
            raise HTTPException(status_code=400, detail="Не указано название теста.")

        loop = asyncio.get_event_loop()
        gift_content = await loop.run_in_executor(None, convert_to_gift, questions)

        user_id = current_user.id
        sanitized_test_name = "".join(c for c in test_name if c.isalnum() or c in (" ", "_", "-")).rstrip()

        file_name = f"{user_id}_{sanitized_test_name}.gift"

        gift_file = io.BytesIO(gift_content.encode("utf-8"))
        gift_file_size = gift_file.getbuffer().nbytes
        gift_file.seek(0)

        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=file_name,
            data=gift_file,
            length=gift_file_size,
            content_type="application/octet-stream"
        )

        return {"message": "Файл сохранен", "test_name": sanitized_test_name}

    except S3Error as s3e:
        raise HTTPException(status_code=500, detail=f"Ошибка MinIO: {str(s3e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла.")


@app.get("/api/tests/", response_model=list[str])
async def get_user_tests(current_user = Depends(get_current_user_dependency)):
    """Получение списка тестов без user_id."""
    user_id = current_user.id

    try:
        objects = minio_client.list_objects(BUCKET_NAME, prefix="", recursive=True)
        user_files = [
            obj.object_name.split("_", 1)[1]  # Убираем user_id из имени файла
            for obj in objects if obj.object_name.startswith(f"{user_id}_")
        ]

        if not user_files:
            raise HTTPException(status_code=404, detail="Тесты не найдены.")

        return user_files

    except S3Error as s3e:
        raise HTTPException(status_code=500, detail=f"Ошибка MinIO: {str(s3e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка при получении списка файлов.")


@app.get("/api/tests/download/{test_name}")
async def download_user_file(
    test_name: str,
    current_user = Depends(get_current_user_dependency)
):
    """Скачивание файла из MinIO (автоматически подставляем user_id)."""
    user_id = current_user.id

    sanitized_test_name = "".join(c for c in test_name if c.isalnum() or c in (" ", "_", "-", ".")).rstrip()
    file_name = f"{user_id}_{sanitized_test_name}.gift"

    try:
        response = minio_client.get_object(BUCKET_NAME, file_name)
        file_data = response.read()
        response.close()
        response.release_conn()

        file_io = io.BytesIO(file_data)
        encoded_filename = urllib.parse.quote(sanitized_test_name)
        content_disposition = f'attachment; filename="{encoded_filename}"'

        return StreamingResponse(
            file_io,
            media_type="application/octet-stream",
            headers={"Content-Disposition": content_disposition}
        )

    except S3Error as s3e:
        if "NoSuchKey" in str(s3e):
            raise HTTPException(status_code=404, detail=f"Файл '{test_name}' не найден.")
        raise HTTPException(status_code=500, detail=f"Ошибка MinIO: {str(s3e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка при скачивании файла.")

@app.get("/api/types/", response_model=List[QuestionTypeResponse])
async def get_all_question_types(
    conn: asyncpg.Connection = Depends(get_db)
):
    """
    Эндпоинт для получения всех типов вопросов.
    """
    try:
        question_types = await get_all_test_types(conn)
        return question_types
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка при получении типов вопросов.")