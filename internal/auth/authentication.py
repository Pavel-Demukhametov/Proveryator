# internal/handlers/authentication.py

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from internal.schemas import UserLogin, Token
from internal.repositories.auth_repository import authenticate_user
import asyncpg

async def handle_login(user: UserLogin, conn: asyncpg.Connection) -> JSONResponse:
    """
    Обработка логина пользователя.
    """
    token = await authenticate_user(conn, user.email, user.password)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return JSONResponse(content=token.dict(), status_code=200)
