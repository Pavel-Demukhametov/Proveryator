# tests/test_user.py

import pytest
from uuid import uuid4
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
        token = "valid.token.string"
        payload = {"sub": "user@example.com"}
        mock_jwt_decode.return_value = payload
        user_id = uuid4()
        user_data = {
            "id": user_id,
            "email": "user@example.com",
            "username": "testuser"
        }
        mock_get_user_by_email.return_value = user_data
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        expected_user = UserResponse(**user_data)

        result = await get_current_user(token=token, conn=mock_conn)

        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_awaited_once_with(mock_conn, "user@example.com")
        assert result == expected_user

    @patch('internal.utils.user.get_user_by_email', new_callable=AsyncMock)
    @patch('internal.utils.user.jwt.decode', side_effect=JWTError("Token decode error"))
    @patch('internal.utils.user.get_db')
    async def test_get_current_user_invalid_token(self, mock_get_db, mock_jwt_decode, mock_get_user_by_email):
        token = "invalid.token.string"
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, conn=mock_conn)

        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_not_called()
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Неверный токен."

    @patch('internal.utils.user.get_user_by_email', new_callable=AsyncMock)
    @patch('internal.utils.user.jwt.decode')
    @patch('internal.utils.user.get_db')
    async def test_get_current_user_no_sub_in_token(self, mock_get_db, mock_jwt_decode, mock_get_user_by_email):
        token = "token.without.sub"
        payload = {}
        mock_jwt_decode.return_value = payload

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, conn=mock_conn)

        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_not_called()
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Неверный токен."

    @patch('internal.utils.user.get_user_by_email', return_value=None)
    @patch('internal.utils.user.jwt.decode')
    @patch('internal.utils.user.get_db')
    async def test_get_current_user_user_not_found(self, mock_get_db, mock_jwt_decode, mock_get_user_by_email):
        token = "valid.token.string"
        payload = {"sub": "nonexistent@example.com"}
        mock_jwt_decode.return_value = payload

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, conn=mock_conn)

        mock_jwt_decode.assert_called_once_with(token, SECRET_KEY, algorithms=[ALGORITHM])
        mock_get_user_by_email.assert_awaited_once_with(mock_conn, "nonexistent@example.com")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Неверный токен."