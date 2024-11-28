from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from internal.models import draft

class Theme(BaseModel):
    id: int
    title: str
    description: str
    isIncluded: bool
    questionCountSingle: Optional[int] = None 
    questionCountOpen: Optional[int] = None    

class TestCreationRequest(BaseModel):
    method: str
    title: str
    totalQuestions: Optional[int] = None
    themes: Optional[List[Theme]] = None
    lectureMaterials: Optional[str] = None

async def handle_lecture_upload(method: str, url: str, file: UploadFile, materials: str):
    print(f"url: {url}")
    print(f"materials: {materials}")
    print(f"file: {file.filename if file else None}")

    if not materials:
        raise HTTPException(
            status_code=400, detail="Материалы лекции обязательны."
        )

    if method == 'YandexDisk' and not url:
        raise HTTPException(
            status_code=400, detail="Для метода YandexDisk требуется ссылка."
        )

    if method == 'Device' and not file:
        raise HTTPException(
            status_code=400, detail="Для метода Device требуется файл."
        )

    response_data = {
        "message": "Лекция успешно обработана!",
        "method": method,
        "url": url if method == 'YandexDisk' else None,
        "file_name": file.filename if file else None,
        "materials": materials,
    }
    return JSONResponse(content=response_data, status_code=200)

async def handle_test_creation(
    method: str, 
    title: str, 
    totalQuestions: Optional[int], 
    themes: Optional[List[Theme]],
    lectureMaterials: Optional[str]
):
    if not title:
        raise HTTPException(status_code=400, detail="Название теста обязательно.")

    if method not in ["general", "byThemes"]:
        raise HTTPException(status_code=400, detail="Неверный метод создания теста.")

    if method == "general":
        if totalQuestions is None or totalQuestions <= 0:
            raise HTTPException(status_code=400, detail="Укажите корректное количество вопросов.")
    elif method == "byThemes":
        if not themes or not isinstance(themes, list):
            raise HTTPException(status_code=400, detail="Темы обязательны при методе byThemes.")
        if len(themes) == 0:
            raise HTTPException(status_code=400, detail="Должна быть выбрана хотя бы одна тема.")
        for theme in themes:
            if theme.isIncluded:
                if theme.questionCountSingle is None or theme.questionCountSingle <= 0:
                    raise HTTPException(status_code=400, detail=f"Укажите корректное количество вопросов с одним правильным ответом для темы '{theme.title}'.")
                if theme.questionCountOpen is None or theme.questionCountOpen <= 0:
                    raise HTTPException(status_code=400, detail=f"Укажите корректное количество вопросов с открытым ответом для темы '{theme.title}'.")
    else:
        parsed_themes = None

    print(f"Метод: {method}")
    print(f"Название: {title}")
    if method == "general":
        print(f"Вопросы: {totalQuestions}")
    elif method == "byThemes":
        print(f"Вопросы по темам:")
        for theme in themes:
            if theme.isIncluded:
                print(f"  Тема: {theme.title}, Количество вопросов с одним правильным ответом: {theme.questionCountSingle}, Количество вопросов с открытым ответом: {theme.questionCountOpen}")
    print(f"Материалы лекции: {lectureMaterials}")

    draft.process_text(lectureMaterials)
    response_data = {
        "message": "Тест успешно создан.",
        "testId": 1, 
        "method": method,
        "title": title,
        "totalQuestions": totalQuestions,
        "themes": [theme.dict() for theme in themes] if themes else None,
        "lectureMaterials": lectureMaterials,
    }

    return JSONResponse(content=response_data, status_code=200)