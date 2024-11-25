from fastapi import FastAPI, Form, File, UploadFile, HTTPException  # Добавлен импорт File и UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
from internal.handlers.handlers import handle_lecture_upload, handle_test_creation, TestCreationRequest
import json



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
    method: str = Form(...),
    url: str = Form(None),
    file: UploadFile = File(None),
    materials: str = Form(...)
):
    return await handle_lecture_upload(method, url, file, materials)

@app.post("/api/tests/create/")
async def test_creation(request: TestCreationRequest):
    return await handle_test_creation(
        method=request.method,
        title=request.title,
        totalQuestions=request.totalQuestions,
        themes=request.themes,
        lectureMaterials=request.lectureMaterials
    )
