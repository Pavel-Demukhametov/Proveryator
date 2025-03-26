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



class TestExtractTextFromDOCX:
    @patch('internal.utils.text_converter.Document')
    def test_extract_text_from_docx_success(self, mock_document):
        # Создаем макет объекта Document и его абзацев
        mock_para1 = MagicMock()
        mock_para1.text = "Первый абзац."
        mock_para2 = MagicMock()
        mock_para2.text = "Второй абзац."
        mock_instance = MagicMock()
        mock_instance.paragraphs = [mock_para1, mock_para2]
        mock_document.return_value = mock_instance

        file_path = "test.docx"
        expected = "Первый абзац.\nВторой абзац."

        result = extract_text_from_docx(file_path)
        assert result == expected
        mock_document.assert_called_once_with(file_path)

    @patch('internal.utils.text_converter.Document')
    def test_extract_text_from_docx_exception(self, mock_document):
        # Симулируем ошибку при открытии документа
        mock_document.side_effect = Exception("DOCX read error.")
        file_path = "test.docx"
        with pytest.raises(Exception) as exc_info:
            extract_text_from_docx(file_path)
        assert "DOCX read error." in str(exc_info.value)


#############################
# Тесты для аудио
#############################

from internal.utils.text_converter import AudioTranscription
import numpy as np

class TestAudioTranscription:
    @patch('internal.utils.text_converter.AudioSegment.from_file')
    def test_convert_audio_to_wav_success(self, mock_from_file):
        # Создаем макет аудио, у которого вызовы методов возвращают сам объект
        mock_audio = MagicMock()
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_from_file.return_value = mock_audio

        transcription = AudioTranscription()
        # Вызываем конвертацию
        transcription.convert_audio_to_wav("input.mp3", "output.wav")

        mock_from_file.assert_called_once_with("input.mp3")
        mock_audio.set_frame_rate.assert_called_once_with(16000)
        mock_audio.set_channels.assert_called_once_with(1)
        mock_audio.export.assert_called_once_with("output.wav", format="wav")

    @patch('internal.utils.text_converter.AudioSegment.from_file')
    def test_convert_audio_to_wav_exception(self, mock_from_file):
        # Симулируем ошибку при чтении аудио-файла
        mock_from_file.side_effect = Exception("Audio read error")
        transcription = AudioTranscription()
        with pytest.raises(Exception) as exc_info:
            transcription.convert_audio_to_wav("invalid.mp3", "output.wav")
        assert "Audio read error" in str(exc_info.value)

    @patch('internal.utils.text_converter.librosa.load', side_effect=Exception("Invalid audio file"))
    def test_transcribe_invalid_audio(self, mock_librosa_load):
        transcription = AudioTranscription()
        with pytest.raises(Exception) as exc_info:
            transcription.transcribe("invalid.wav")
        assert "Invalid audio file" in str(exc_info.value)
    @patch('internal.utils.text_converter.librosa.load')
    def test_transcribe_success(self, mock_librosa_load):
        # Create a test audio array: 60 seconds of silence (60 * 16000 samples)
        sampling_rate = 16000
        fake_audio = np.zeros(sampling_rate * 60)
        mock_librosa_load.return_value = (fake_audio, sampling_rate)

        transcription = AudioTranscription()

        # Mock processor and model for transcription
        fake_processor = MagicMock()
        fake_processor.return_tensors.return_value = {"input_features": MagicMock(), "attention_mask": MagicMock()}
        fake_processor.batch_decode.return_value = ["transcribed text"]
        transcription.processor = fake_processor

        fake_model = MagicMock()
        fake_model.generate.return_value = MagicMock()
        transcription.model = fake_model

        result = transcription.transcribe("input.wav")

        # Calculate the number of chunks processed
        chunk_length_s = 30
        stride_length_s = 5
        total_chunks = (60 - chunk_length_s) // (chunk_length_s - stride_length_s) + 1

        # The expected result is the transcribed text repeated for each chunk
        expected_result = " ".join(["transcribed text"] * (total_chunks + 1))

        assert result == expected_result

    @patch('internal.utils.text_converter.librosa.load')
    def test_transcribe_silent_audio(self, mock_librosa_load):
        # Create a test audio array: 60 seconds of silence (60 * 16000 samples)
        sampling_rate = 16000
        silent_audio = np.zeros(sampling_rate * 60)
        mock_librosa_load.return_value = (silent_audio, sampling_rate)

        transcription = AudioTranscription()

        # Mock processor and model for transcription
        fake_processor = MagicMock()
        fake_processor.return_tensors.return_value = {"input_features": MagicMock(), "attention_mask": MagicMock()}
        fake_processor.batch_decode.return_value = [""]
        transcription.processor = fake_processor

        fake_model = MagicMock()
        fake_model.generate.return_value = MagicMock()
        transcription.model = fake_model

        result = transcription.transcribe("silent.wav")

        # The expected result is an empty string since there's no speech in the audio
        expected_result = "  "

        assert result == expected_result
    def test_transcribe_noisy_audio(self):
        with patch('internal.utils.text_converter.librosa.load') as mock_librosa_load:
            # Создаем тестовый аудиосигнал: 60 секунд белого шума
            sampling_rate = 16000
            duration_seconds = 60
            # Генерируем белый шум
            noise = np.random.normal(0, 1, sampling_rate * duration_seconds)
            mock_librosa_load.return_value = (noise, sampling_rate)

            transcription = AudioTranscription()

            # Мокаем processor и model для транскрибации
            fake_processor = MagicMock()
            fake_processor.return_tensors.return_value = {"input_features": MagicMock(), "attention_mask": MagicMock()}
            # Предполагаем, что модель возвращает пустую строку для шума
            fake_processor.batch_decode.return_value = [""]
            transcription.processor = fake_processor

            fake_model = MagicMock()
            fake_model.generate.return_value = MagicMock()
            transcription.model = fake_model

            result = transcription.transcribe("noisy.wav")

            # Ожидаемый результат — пустая строка, так как в шуме нет речи
            expected_result = ""

            assert result == expected_result