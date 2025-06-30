import asyncio
import asyncpg
import io
import os
import sys
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from urllib.parse import quote

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    status,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from minio import Minio
from minio.error import S3Error
from pydantic import ValidationError
import jwt
from jwt import PyJWTError

from internal.auth.authentication import handle_login
from internal.auth.registration import handle_registration
from internal.database import get_db
from internal.handlers.handlers import handle_lecture_upload, router as lecture_router
from internal.handlers.handlers import _run_generation
from internal.repositories.auth_repository import ALGORITHM, SECRET_KEY
from internal.repositories.user_repository import get_user_by_email
from internal.schemas import (
    ByThemesTestCreationRequest,
    GeneralTestCreationRequest,
    TestCreationRequest,
    TestCreationResponse,
    Token,
    TokenData,
    UserCreate,
    UserLogin,
    UserResponse,
)
from internal.term_extractor.term_extractor import TermExtractor
from internal.utils.gift_generation import convert_to_gift
from internal.utils.user import get_current_user

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

minio_endpoint = os.getenv("MINIO_ENDPOINT")
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
minio_secure = os.getenv("MINIO_SECURE", "False").lower() == "true"
minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")

if not all([minio_endpoint, minio_access_key, minio_secret_key, minio_bucket_name]):
    logger.error("Missing required MinIO environment variables (MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME)")
    raise ValueError("One or more required MinIO environment variables are not set")
try:
    minio_client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=minio_secure
    )
    logger.info(f"MinIO client initialized with endpoint: {minio_endpoint}")
except Exception as e:
    logger.error(f"Failed to initialize MinIO client: {e}")
    raise

BUCKET_NAME = minio_bucket_name
app.include_router(lecture_router, prefix="/api")

progress_map: dict[str, float] = {}
result_map: dict[str, TestCreationResponse] = {}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login/")
keyword_extractor = TermExtractor()
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
    """Аутентификация и получение текущего пользователя на основе предоставленного JWT-токена."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt_decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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


@app.post("/api/register/", status_code=201)
async def register_user(request: Request, conn: asyncpg.Connection = Depends(get_db)):
    """Регистрация нового пользователя."""
    try:
        user_data = await request.json()
        user = UserCreate(**user_data)
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail="Validation error")
    except Exception as e:
        logger.error(f"Invalid request format: {e}")
        raise HTTPException(status_code=400, detail="Invalid request format")
    return await handle_registration(user, conn)
    
@app.post("/api/login/", response_model=Token)
async def login(user: UserLogin, conn: asyncpg.Connection = Depends(get_db)):
    """Логин пользователя и получение JWT токена."""
    return await handle_login(user, conn)

@app.post("/api/upload/")
async def lecture_upload(
    files: Optional[List[UploadFile]] = File(None),
    materials: Optional[str] = Form(None)
):
    """Загрузка лекционных материалов и файлов для обработки."""
    return await handle_lecture_upload(files, materials)

@app.post("/api/tests/download/")
async def download_gift_file(request: Request):
    """Скачивание файла GIFT на основе предоставленных вопросов и названия теста."""
    try:
        data = await request.json()
        test_title = data.get("title", "default_test").strip()
        if not test_title:
            raise HTTPException(status_code=400, detail="Test title not provided")
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
        logger.error(f"Error creating GIFT file: {e}")
        raise HTTPException(status_code=500, detail="Error creating GIFT file")

@app.post("/api/tests/create/", status_code=202)
async def test_creation(
    request: Request,
    background_tasks: BackgroundTasks,
    conn: asyncpg.Connection = Depends(get_db),
):
    """Создание теста с использованием указанного метода (general или byThemes)."""
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON format")

    method = raw.get("method")
    if method not in ("general", "byThemes"):
        raise HTTPException(400, "Invalid test creation method")
    try:
        if method == "general":
            test_request = GeneralTestCreationRequest(**raw)
        else:
            test_request = ByThemesTestCreationRequest(**raw)
    except ValidationError as ve:
        raise HTTPException(400, "Validation error")

    task_id = str(uuid.uuid4())
    progress_map[task_id] = 0.0
    result_map[task_id] = None  

    background_tasks.add_task(_run_generation, task_id, test_request, progress_map, result_map)

    return {"task_id": task_id}

@app.get("/api/tests/{task_id}/progress")
async def get_test_progress(task_id: str):
    """
    Получение процента выполнения задач. Возвращает JSON: { "progress": 0.0–1.0 }.
    """
    if task_id not in progress_map:
        raise HTTPException(404, "Task not found")
    return {"progress": progress_map[task_id]}


@app.get("/api/tests/{task_id}/result", response_model=TestCreationResponse)
async def get_test_result(task_id: str):
    """
    Получение результата задачи. Если прогресс <1.0 — 202, иначе отдает готовый TestCreationResponse.
    """
    if task_id not in progress_map:
        raise HTTPException(404, "Task not found")
    if progress_map[task_id] < 1.0:
        raise HTTPException(202, "Result not ready")
    return result_map[task_id]

@app.post("/api/tests/save/")
async def test_save(
    request: Request,
    current_user = Depends(get_current_user_dependency)
):
    """Сохранение теста в MinIO"""
    try:
        data = await request.json()
        test_name = data.get("test_name", "default_test").strip()
        questions = data.get("questions", [])

        if not test_name:
            raise HTTPException(status_code=400, detail="Test name not provided")

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

        return {"message": "File saved", "test_name": sanitized_test_name}

    except S3Error as s3e:
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(s3e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving file")

@app.get("/api/tests/", response_model=list[str])
async def get_user_tests(current_user = Depends(get_current_user_dependency)):
    """Получение списка тестов без user_id."""
    user_id = current_user.id

    try:
        objects = minio_client.list_objects(BUCKET_NAME, prefix="", recursive=True)
        user_files = [
            obj.object_name.split("_", 1)[1] 
            for obj in objects if obj.object_name.startswith(f"{user_id}_")
        ]

        if not user_files:
            raise HTTPException(status_code=404, detail="Tests not found")

        return user_files

    except S3Error as s3e:
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(s3e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving file list")

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
            raise HTTPException(status_code=404, detail=f"File '{test_name}' not found")
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(s3e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error downloading file")