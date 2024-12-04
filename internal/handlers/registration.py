# internal/handlers/registration.py

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from internal.schemas import UserCreate, UserResponse
from internal.repositories.user_repository import (
    get_user_by_username,
    get_user_by_email,
    create_user
)
import asyncpg

async def handle_registration(user: UserCreate, conn: asyncpg.Connection) -> JSONResponse:
    """
    Обработка регистрации нового пользователя.
    """
    existing_user = await get_user_by_username(conn, user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username уже занят.")

    existing_email = await get_user_by_email(conn, user.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email уже зарегистрирован.")
    new_user = await create_user(conn, user)

    return JSONResponse(content=new_user.dict(), status_code=201)
