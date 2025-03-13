# internal/repositories/test_types_repository.py

import asyncpg
from typing import List, Optional, Dict
from fastapi import HTTPException
from internal.schemas import QuestionTypeResponse 

async def get_all_test_types(conn: asyncpg.Connection) -> List[Dict]:
    """
    Получает все типы теста из базы данных.
    
    Выполняет запрос к таблице content_types и возвращает список типов в виде словарей.
    """
    try:
        rows = await conn.fetch("SELECT * FROM content_types ORDER BY name")
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка при получении типов теста.")

async def get_test_type_by_id(conn: asyncpg.Connection, type_id: str) -> Optional[Dict]:
    """
    Получает тип теста по его идентификатору.
    
    Принимает идентификатор типа теста и возвращает объект типа словаря, если запись найдена, или None.
    """
    try:
        row = await conn.fetchrow("SELECT * FROM test_types WHERE id = $1", type_id)
        return dict(row) if row else None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка при получении типа теста.")
