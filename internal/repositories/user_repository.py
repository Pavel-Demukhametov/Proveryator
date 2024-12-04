# internal/repositories/user_repository.py

import asyncpg
from typing import Optional, Dict
from internal.schemas import UserCreate, UserResponse
from argon2 import PasswordHasher
from fastapi import HTTPException

ph = PasswordHasher()

async def get_user_by_username(conn: asyncpg.Connection, username: str) -> Optional[Dict]:
    """
    Получает пользователя по username.
    """
    user = await conn.fetchrow("SELECT * FROM users WHERE username = $1", username)
    return dict(user) if user else None

async def get_user_by_email(conn: asyncpg.Connection, email: str) -> Optional[Dict]:
    """
    Получает пользователя по email.
    """
    user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
    return dict(user) if user else None

async def create_user(conn: asyncpg.Connection, user: UserCreate) -> UserResponse:
    """
    Создает нового пользователя.
    """

    hashed_password = ph.hash(user.password)

    new_user = await conn.fetchrow(
        """
        INSERT INTO users (username, email, hashed_password)
        VALUES ($1, $2, $3)
        RETURNING id, username, email
        """,
        user.username,
        user.email,
        hashed_password
    )

    if not new_user:
        raise HTTPException(status_code=500, detail="Не удалось создать пользователя.")


    return UserResponse(
        id=new_user["id"],
        username=new_user["username"],
        email=new_user["email"]
    )
