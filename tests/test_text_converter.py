# tests/test_text_converter.py

import pytest
from unittest.mock import patch, mock_open, MagicMock
from internal.utils.text_converter import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt

class TestExtractTextFromPDF:
    @patch('internal.utils.text_converter.PdfReader')
    def test_extract_text_from_pdf_success(self, mock_pdf_reader):
        # Создание мок-страниц
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 text."
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 text."
        
        # Настройка атрибута pages
        mock_pdf_reader.return_value.pages = [mock_page1, mock_page2]
        
        file_path = "test.pdf"
        expected = "Page 1 text.Page 2 text."
        
        result = extract_text_from_pdf(file_path)
        
        assert result == expected
        mock_pdf_reader.assert_called_once_with(file_path)
        mock_page1.extract_text.assert_called_once()
        mock_page2.extract_text.assert_called_once()
    
    @patch('internal.utils.text_converter.PdfReader')
    def test_extract_text_from_pdf_exception(self, mock_pdf_reader):
        # Настройка мока для генерации исключения
        mock_pdf_reader.side_effect = Exception("PDF read error.")
        
        file_path = "test.pdf"
        with pytest.raises(Exception) as exc_info:
            extract_text_from_pdf(file_path)
        assert "PDF read error." in str(exc_info.value)



class TestExtractTextFromTXT:
    def test_extract_text_from_txt_success(self):
        file_path = "test.txt"
        file_content = "This is a test text."
        with patch('builtins.open', mock_open(read_data=file_content)) as mock_file:
            result = extract_text_from_txt(file_path)
            assert result == file_content
            mock_file.assert_called_once_with(file_path, 'r', encoding='utf-8')
    
    def test_extract_text_from_txt_exception(self):
        file_path = "test.txt"
        with patch('builtins.open', side_effect=Exception("TXT read error.")):
            with pytest.raises(Exception) as exc_info:
                extract_text_from_txt(file_path)
            assert "TXT read error." in str(exc_info.value)
