from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

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

    # Ответ
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
    totalQuestions: int, 
    themes: str, 
    lectureMaterials: str
):
    if not title:
        raise HTTPException(status_code=400, detail="Название теста обязательно.")

    if method not in ["general", "byThemes"]:
        raise HTTPException(status_code=400, detail="Неверный метод создания теста.")

    if method == "general" and (not totalQuestions or totalQuestions <= 0):
        raise HTTPException(status_code=400, detail="Укажите корректное количество вопросов.")

    if method == "byThemes":
        try:
            parsed_themes = json.loads(themes)
            if not isinstance(parsed_themes, list) or not parsed_themes:
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            raise HTTPException(status_code=400, detail="Темы должны быть валидным JSON массивом.")
    else:
        parsed_themes = None

    # Логика создания теста
    print(f"Метод: {method}")
    print(f"Название: {title}")
    print(f"Вопросы: {totalQuestions if method == 'general' else parsed_themes}")
    print(f"Материалы лекции: {lectureMaterials}")

    # Пример ответа
    response_data = {
        "message": "Тест успешно создан.",
        "test_id": 1,  # Идентификатор теста
        "method": method,
        "title": title,
        "totalQuestions": totalQuestions,
        "themes": parsed_themes,
        "lectureMaterials": lectureMaterials,
    }

    return JSONResponse(content=response_data, status_code=200)
