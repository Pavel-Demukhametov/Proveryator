# main.py

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
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
from internal.database import get_db
import json
from fastapi.responses import JSONResponse
import logging
from pydantic import ValidationError
import asyncpg
from jose import JWTError, jwt
from typing import Optional
from fastapi.security import OAuth2PasswordBearer
from internal.repositories.user_repository import get_user_by_email
from internal.repositories.auth_repository import SECRET_KEY, ALGORITHM

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login/")

async def get_current_user(token: str = Depends(oauth2_scheme), conn: asyncpg.Connection = Depends(get_db)) -> UserResponse:
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

@app.post("/api/register/", status_code=201)
async def register_user(request: Request, conn: asyncpg.Connection = Depends(get_db)):
    """
    Регистрация нового пользователя.
    """
    logger.info("Получен запрос на регистрацию пользователя")

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
    file: UploadFile = File(...),
    materials: str = Form(...)
):
    logger.info("Получен запрос на загрузку лекции")
    return await handle_lecture_upload(file, materials)

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
            try:
                test_request = ByThemesTestCreationRequest(**data)
            except ValidationError as ve:
                logger.error(f"Валидационная ошибка: {ve}")
                raise HTTPException(status_code=400, detail=ve.errors())
        else:
            raise HTTPException(status_code=400, detail="Неверный метод создания теста.")

        logger.info(f"Method: {test_request.method}")
        logger.info(f"Title: {test_request.title}")
        if isinstance(test_request, GeneralTestCreationRequest):
            logger.info(f"Multiple Choice Count: {test_request.multipleChoiceCount}")
            logger.info(f"Open Answer Count: {test_request.openAnswerCount}")
        elif isinstance(test_request, ByThemesTestCreationRequest):
            logger.info("Themes:")
            for theme in test_request.themes:
                logger.info(f"  - Keyword: {theme.keyword}")
                logger.info(f"    Sentences: {theme.sentences}")
                if isinstance(theme, ByThemesTheme):
                    logger.info(f"    Multiple Choice Count: {theme.multipleChoiceCount}")
                    logger.info(f"    Open Answer Count: {theme.openAnswerCount}")
        logger.info(f"Lecture Materials: {test_request.lectureMaterials}")
        return await handle_test_creation(test_request)

    except json.JSONDecodeError:
        logger.error("Некорректный формат JSON.")
        raise HTTPException(status_code=400, detail="Некорректный формат JSON.")
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=400, detail=str(e))
