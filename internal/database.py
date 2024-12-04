# internal/database.py

import asyncpg
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение URL базы данных из переменных окружения или использование дефолтного значения
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/Proveryator")

async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Зависимость для получения соединения с базой данных.
    """
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()
