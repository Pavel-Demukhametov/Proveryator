from fastapi import FastAPI, Form, File, UploadFile  # Добавлен импорт File и UploadFile
from fastapi.middleware.cors import CORSMiddleware
from internal.handlers.handlers import handle_lecture_upload, handle_test_creation

app = FastAPI()

# Добавление CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить доступ с любых доменов
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

@app.post("/api/upload/")
async def lecture_upload(
    method: str = Form(...),  # Указываем, что параметр будет передан через форму
    url: str = Form(None),   # Параметр URL передаётся через форму (опционально)
    file: UploadFile = File(None),  # Файл (опционально)
    materials: str = Form(...)  # Материалы лекции (обязательно)
):
    return await handle_lecture_upload(method, url, file, materials)

@app.post("/api/tests/create/")
async def test_creation(
    method: str = Form(...),
    title: str = Form(...),
    totalQuestions: int = Form(None),
    themes: str = Form(None),  # JSON-строка с темами
    lectureMaterials: str = Form(None)
):
    return await handle_test_creation(method, title, totalQuestions, themes, lectureMaterials)