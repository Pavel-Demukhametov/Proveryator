from pydantic import BaseModel, EmailStr, Field, model_validator
from typing import List, Optional, Literal, Union
from uuid import UUID

class GeneralTheme(BaseModel):
    keyword: str
    sentences: List[str]  

class ByThemesTheme(BaseModel):
    keyword: str
    multipleChoiceCount: int = Field(..., ge=0, description="Количество вопросов с одним правильным ответом.")
    openAnswerCount: int = Field(..., ge=0, description="Количество вопросов с открытым ответом.")

class GeneralTestCreationRequest(BaseModel):
    method: Literal['general']
    title: str
    multipleChoiceCount: Optional[int] = Field(
        0, ge=0, description="Общее количество вопросов с одним правильным ответом."
    )
    openAnswerCount: Optional[int] = Field(
        0, ge=0, description="Общее количество вопросов с открытым ответом."
    )
    lectureMaterials: str
    themes: List[GeneralTheme] = Field(
        ..., description="Список тем для метода general."
    )

    @model_validator(mode='after')
    def check_question_counts(self):
        mc = self.multipleChoiceCount or 0
        oa = self.openAnswerCount or 0
        if mc <= 0 and oa <= 0:
            raise ValueError(
                'Необходимо указать хотя бы одно количество вопросов: с одним правильным ответом или с открытым ответом.'
            )
        return self
class Theme(BaseModel):
    keyword: Union[str, List[str]] = Field(..., alias="keyword")
    multipleChoiceCount: int = Field(0, description="Количество вопросов с одним правильным ответом.")
    openAnswerCount: int = Field(0, description="Количество вопросов с открытым ответом.")

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True
class ByThemesTestCreationRequest(BaseModel):
    method: str
    title: str
    lectureMaterials: str
    themes: List[Theme]


TestCreationRequest = Union[GeneralTestCreationRequest, ByThemesTestCreationRequest]

class Question(BaseModel):
    type: str 
    question: str
    answer: Optional[str] = None 
    options: Optional[List[str]] = None  
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
    id: UUID
    username: str
    email: str

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

class QuestionTypeResponse(BaseModel):
    id: UUID
    name: str

    class Config:
        orm_mode = True