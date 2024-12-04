# internal/repositories/auth_repository.py

import asyncpg
from internal.schemas import UserLogin, Token, TokenData
from internal.repositories.user_repository import get_user_by_email
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from jose import JWTError, jwt
from fastapi import HTTPException, status
from datetime import datetime, timedelta
from typing import Optional
import os


ph = PasswordHasher()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Проверяет соответствие пароля с хэшем.
    """
    try:
        ph.verify(hashed_password, plain_password)
        return True
    except VerifyMismatchError:
        return False

def get_password_hash(password: str) -> str:
    """
    Хэширует пароль.
    """
    return ph.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Создает JWT токен доступа.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def authenticate_user(conn: asyncpg.Connection, email: str, password: str) -> Optional[Token]:
    """
    Аутентифицирует пользователя и возвращает JWT токен доступа.
    """
    user = await get_user_by_email(conn, email)
    if not user:
        return None
    if not verify_password(password, user['hashed_password']):
        return None
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['email']},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")
