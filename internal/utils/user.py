# internal/utils/user.py

from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional
import asyncpg
from internal.repositories.auth_repository import SECRET_KEY, ALGORITHM
from internal.repositories.user_repository import get_user_by_email
from internal.database import get_db  # Убедитесь, что путь правильный
  
# Определение oauth2_scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

from internal.schemas import UserResponse  # Убедитесь, что импортируете модель

async def get_current_user(token: str = Depends(oauth2_scheme), conn: asyncpg.Connection = Depends(get_db)) -> UserResponse:
    """
    Получаем текущего пользователя на основе JWT токена.
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Неверный токен.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
        user = await get_user_by_email(conn, email)
        if user is None:
            raise credentials_exception
        return UserResponse(**user)  # Преобразуем dict в Pydantic модель
    except JWTError:
        raise credentials_exception