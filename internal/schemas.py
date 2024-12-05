# internal/schemas.py

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Literal, Union



class GeneralTheme(BaseModel):
    keyword: str
    sentences: List[str]  

class ByThemesTheme(BaseModel):
    keyword: str
    sentences: List[str]
    multipleChoiceCount: int = Field(..., ge=1, description="Количество вопросов с одним правильным ответом.")
    openAnswerCount: int = Field(..., ge=1, description="Количество вопросов с открытым ответом.")

class Theme(BaseModel):
    keyword: str
    sentences: List[str]
    multipleChoiceCount: Optional[int] = 0
    openAnswerCount: Optional[int] = 0


class GeneralTestCreationRequest(BaseModel):
    method: Literal['general']
    title: str
    multipleChoiceCount: int = Field(..., ge=1, description="Общее количество вопросов с одним правильным ответом.")
    openAnswerCount: int = Field(..., ge=1, description="Общее количество вопросов с открытым ответом.")
    lectureMaterials: str
    themes: List[GeneralTheme] = Field(..., description="Список тем для метода general.")


class ByThemesTestCreationRequest(BaseModel):
    method: Literal['byThemes']
    title: str
    lectureMaterials: str
    themes: List[ByThemesTheme] = Field(..., min_items=1, description="Должна быть выбрана хотя бы одна тема.")



TestCreationRequest = Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]


class Question(BaseModel):
    type: str  # "mc" или "open"
    question: str
    answer: Optional[str] = None  # Правильный ответ для "open" вопросов
    options: Optional[List[str]] = None  # Варианты ответов для "mc" вопросов
    sentence: str

    class Config:
        orm_mode = True



class TestCreationResponse(BaseModel):
    message: str
    method: str
    title: str
    lectureMaterials: str
    questions: List[Question]
    themes: List[Theme]


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Имя пользователя должно быть от 3 до 50 символов.")
    email: EmailStr
    password: str = Field(..., min_length=6, description="Пароль должен содержать минимум 6 символов.")


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

    class Config:
        orm_mode = True


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
