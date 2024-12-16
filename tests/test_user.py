# tests/test_user.py

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException
from internal.utils.user import get_current_user
from internal.repositories.auth_repository import SECRET_KEY, ALGORITHM
from internal.schemas import UserResponse
from jose import JWTError

@pytest.mark.asyncio
class TestGetCurrentUser:
    @patch('internal.utils.user.get_user_by_email', new_callable=AsyncMock)
    @patch('internal.utils.user.jwt.decode')
    @patch('internal.utils.user.get_db')
    async def test_get_current_user_success(self, mock_get_db, mock_jwt_decode, mock_get_user_by_email):
        # Настройка мока для jwt.decode
        token = "valid.token.string"
        payload = {"sub": "user@example.com"}
        mock_jwt_decode.return_value = payload

        # Настройка мока для get_user_by_email
        user_data = {
            "id": 1,
            "email": "user@example.com",
            "name": "Test User",
            "username": "testuser"  # Добавлено поле username
        }
        mock_get_user_by_email.return_value = user_data

        # Настройка мока для get_db (возвращает мок соединения)
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        # Ожидаемый результат
        expected_user = UserResponse(**user_data)

        # Вызов функции
        result = await get_current_user(token=token, conn=mock_conn)

        # Проверки
        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_awaited_once_with(mock_conn, "user@example.com")
        assert result == expected_user

    @patch('internal.utils.user.get_user_by_email', new_callable=AsyncMock)
    @patch('internal.utils.user.jwt.decode', side_effect=JWTError("Token decode error"))
    @patch('internal.utils.user.get_db')
    async def test_get_current_user_invalid_token(self, mock_get_db, mock_jwt_decode, mock_get_user_by_email):
        # Настройка мока для jwt.decode (генерирует ошибку)
        token = "invalid.token.string"

        # Настройка мока для get_db (возвращает мок соединения)
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        # Вызов функции и ожидание исключения
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, conn=mock_conn)

        # Проверки
        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_not_called()
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Неверный токен."

    @patch('internal.utils.user.get_user_by_email', new_callable=AsyncMock)
    @patch('internal.utils.user.jwt.decode')
    @patch('internal.utils.user.get_db')
    async def test_get_current_user_no_sub_in_token(self, mock_get_db, mock_jwt_decode, mock_get_user_by_email):
        # Настройка мока для jwt.decode (отсутствует "sub")
        token = "token.without.sub"
        payload = {}
        mock_jwt_decode.return_value = payload

        # Настройка мока для get_db (возвращает мок соединения)
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        # Вызов функции и ожидание исключения
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, conn=mock_conn)

        # Проверки
        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_not_called()
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Неверный токен."

    @patch('internal.utils.user.get_user_by_email', return_value=None)
    @patch('internal.utils.user.jwt.decode')
    @patch('internal.utils.user.get_db')
    async def test_get_current_user_user_not_found(self, mock_get_db, mock_jwt_decode, mock_get_user_by_email):
        # Настройка мока для jwt.decode
        token = "valid.token.string"
        payload = {"sub": "nonexistent@example.com"}
        mock_jwt_decode.return_value = payload

        # Настройка мока для get_db (возвращает мок соединения)
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        # Вызов функции и ожидание исключения
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, conn=mock_conn)

        # Проверки
        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_awaited_once_with(mock_conn, "nonexistent@example.com")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Неверный токен."
