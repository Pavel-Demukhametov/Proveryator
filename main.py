# main.py

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from internal.handlers.handlers import handle_lecture_upload, handle_test_creation
from internal.handlers.registration import handle_registration
from internal.handlers.authentication import handle_login
from internal.schemas import (
    TestCreationRequest,
    ByThemesTheme,
    GeneralTestCreationRequest,
    ByThemesTestCreationRequest,
    UserCreate,
    UserResponse,
    UserLogin,
    Token,
    TokenData
)
from internal.keywords_extractor import KeywordsExtractor
from typing import List, Optional, Union
from internal.database import get_db
import json
from pydantic import ValidationError
import asyncpg
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
from internal.utils.user import get_current_user
from internal.utils.gift_generation import convert_to_gift
from internal.repositories.auth_repository import SECRET_KEY, ALGORITHM
from internal.repositories.user_repository import get_user_by_email
import os
from urllib.parse import quote
import io
import sys
import logging
import urllib.parse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Настройте согласно требованиям безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login/")
keywords_extractor = KeywordsExtractor()
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if hasattr(handler, 'setEncoding'):
    handler.setEncoding('utf-8')

logger.addHandler(handler)

# Создайте глобальный ThreadPoolExecutor для CPU-емких задач
executor = ThreadPoolExecutor(max_workers=8)  # Настройте количество потоков в зависимости от нагрузки

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
    except JWTError:
        raise credentials_exception
    user = await get_user_by_email(conn, token_data.email)
    if user is None:
        raise credentials_exception
    return UserResponse(id=user["id"], username=user["username"], email=user["email"])

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
        logger.info(json.dumps(user_data, ensure_ascii=False, indent=4))
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
    logger.info("Получен запрос на логин пользователя:")
    logger.info(json.dumps(user.dict(), ensure_ascii=False, indent=4))
    return await handle_login(user, conn)

@app.post("/api/upload/")
async def lecture_upload(
    file: Optional[UploadFile] = File(None),
    materials: Optional[str] = Form(None)
):
    logger.info("Получен запрос на загрузку лекции")
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

        # Санитизация названия теста
        sanitized_test_title = "".join(c for c in test_title if c.isalnum() or c in (" ", "_", "-")).rstrip()
        file_name = f"{sanitized_test_title}.gift"
        encoded_file_name = quote(file_name.encode('utf-8'))
        # Получаем вопросы и материалы лекций из запроса
        questions = data.get("questions", [])
        lecture_materials = data.get("lectureMaterials", [])

        # Конвертация данных в формат GIFT в отдельном потоке
        gift_content = await asyncio.get_event_loop().run_in_executor(
            executor, convert_to_gift, questions
        )
        logger.debug("Конвертация в GIFT выполнена успешно.")

        # Создаём временный файл в памяти
        gift_file = io.BytesIO(gift_content.encode("utf-8"))

        # URL encode the file name to handle special characters
        encoded_filename = urllib.parse.quote(file_name)

        # Prepare the content-disposition header
        content_disposition = f"attachment; filename=\"{encoded_filename}\""

        logger.info(f"Отправка файла: {file_name}")

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
        logger.info("Получен запрос на создание теста:")
        logger.info(json.dumps(data, ensure_ascii=False, indent=4))

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
                keyword_lemmatized = keywords_extractor.tokenize_lemmatize(keyword)
                sentences_with_keyword = keywords_extractor.extract_sentences_with_keyword(combined_materials, keyword_lemmatized)

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
    # Создайте новый event loop для выполнения асинхронных задач внутри синхронного контекста
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
    conn: asyncpg.Connection = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user_dependency)
):
    """
    Сохранение теста для зарегистрированного пользователя.
    Файл сохраняется под именем, состоящим из user_id и названия теста.
    """
    try:
        data = await request.json()
        logger.info("Получен запрос на сохранение теста:")
        logger.info(json.dumps(data, ensure_ascii=False, indent=4))

        # Извлекаем название теста из данных
        test_name = data.get("test_name", "default_test").strip()
        questions = data.get("questions", [])
        lecture_materials = data.get("lectureMaterials", [])

        if not test_name:
            raise HTTPException(status_code=400, detail="Не указано название теста.")

        # Конвертируем данные в формат GIFT в отдельном потоке
        gift_content = await asyncio.get_event_loop().run_in_executor(
            executor, convert_to_gift, questions
        )
        logger.debug("Конвертация в GIFT выполнена успешно.")

        # Формируем имя файла с учетом user_id и названия теста
        user_id = current_user.id  # Доступ через атрибуты
        if not user_id:
            raise HTTPException(status_code=400, detail="Не удалось определить пользователя.")

        # Очистка названия теста для использования в имени файла
        sanitized_test_name = "".join(c for c in test_name if c.isalnum() or c in (" ", "_", "-")).rstrip()
        file_name = f"{user_id}_{sanitized_test_name}.gift"
        gift_file_path = os.path.join("gift_files", file_name)

        # Асинхронное создание директорий и запись файла
        os.makedirs(os.path.dirname(gift_file_path), exist_ok=True)
        async with aiofiles.open(gift_file_path, "w", encoding="utf-8") as file:
            await file.write(gift_content)
        logger.info(f"GIFT файл сохранён как {gift_file_path}")

        # Создаём временный файл в памяти
        gift_file = io.BytesIO(gift_content.encode("utf-8"))

        # URL-кодирование имени файла для заголовка Content-Disposition
        encoded_filename = urllib.parse.quote(file_name)

        # Устанавливаем заголовок Content-Disposition с поддержкой UTF-8
        content_disposition = f"attachment; filename=\"{encoded_filename}\""

        return StreamingResponse(
            gift_file,
            media_type="application/octet-stream",
            headers={"Content-Disposition": content_disposition}
        )

    except HTTPException as he:
        # HTTPException уже содержит нужный статус и detail
        raise he
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла GIFT: {e}")
        raise HTTPException(status_code=500, detail="Произошла ошибка при сохранении файла GIFT.")

@app.get("/api/tests/", response_model=List[str])
async def get_user_tests(current_user: UserResponse = Depends(get_current_user_dependency)):
    """
    Endpoint для получения списка тестов, созданных пользователем.
    Возвращает имена всех файлов, содержащих ID пользователя в названии.
    """
    user_id = current_user.id

    # Асинхронное чтение каталога
    try:
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, os.listdir, "gift_files")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Каталог с файлами не найден.")

    # Фильтруем файлы, оставляя только те, у которых ID пользователя в имени
    user_files = [file for file in files if str(user_id) in file]

    if not user_files:
        raise HTTPException(status_code=404, detail="Тесты не найдены для данного пользователя.")

    # Возвращаем список имен файлов
    return user_files

# Route to handle test file downloads
@app.get("/api/tests/download/{file_name}")
async def download_user_file(file_name: str, current_user: UserResponse = Depends(get_current_user_dependency)):
    """
    Эндпоинт для скачивания файла по его названию.
    Принимает название файла, если файл существует и принадлежит пользователю, возвращает его для скачивания.
    """
    user_id = current_user.id

    logger.info(f"Пользователь {current_user.email} запрашивает файл: {file_name}")

    # Проверяем, что имя файла начинается с user_id
    if not file_name.startswith(f"{user_id}_"):
        logger.warning(f"Доступ запрещён для файла: {file_name}")
        raise HTTPException(status_code=403, detail="Доступ запрещён.")

    # Санитизация имени файла
    sanitized_file_name = "".join(c for c in file_name if c.isalnum() or c in (" ", "_", "-", ".")).rstrip()

    # Проверяем, что после санитизации имя файла всё ещё начинается с user_id
    if not sanitized_file_name.startswith(f"{user_id}_"):
        logger.warning(f"Доступ запрещён после санитизации для файла: {file_name}")
        raise HTTPException(status_code=403, detail="Доступ запрещён.")

    # Проверяем, что файл имеет расширение .gift
    if not sanitized_file_name.endswith(".gift"):
        logger.warning(f"Неверный формат файла: {sanitized_file_name}")
        raise HTTPException(status_code=400, detail="Неверный формат файла.")

    file_path = os.path.join("gift_files", sanitized_file_name)

    # Асинхронная проверка существования файла
    exists = await asyncio.get_event_loop().run_in_executor(None, os.path.exists, file_path)
    if not exists:
        logger.error(f"Файл не найден: {file_path}")
        raise HTTPException(status_code=404, detail=f"Файл с названием {file_name} не найден.")

    try:
        # Асинхронное чтение файла
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        # Создаём объект BytesIO
        file_io = io.BytesIO(content)

        # URL кодирование имени файла для корректной передачи
        encoded_filename = urllib.parse.quote(sanitized_file_name)

        # Подготовка заголовка Content-Disposition
        content_disposition = f"attachment; filename=\"{encoded_filename}\""

        logger.info(f"Файл {sanitized_file_name} успешно подготовлен для скачивания.")

        # Возвращаем файл как StreamingResponse
        return StreamingResponse(
            file_io,
            media_type="application/octet-stream",
            headers={"Content-Disposition": content_disposition}
        )

    except Exception as e:
        logger.error(f"Ошибка при скачивании файла {file_name}: {e}")
        raise HTTPException(status_code=500, detail="Произошла ошибка при скачивании файла.")
