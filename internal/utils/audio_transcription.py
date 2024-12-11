
import os
import logging
from typing import Optional
import ffmpeg
from pydub import AudioSegment
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa  # Добавлено
from pydub import AudioSegment

# Явно указываем путь к FFmpeg (если необходимо)
AudioSegment.converter = "D:\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe" 
logger = logging.getLogger(__name__)

class AudioTranscription:
    def __init__(self, model_name: str = "openai/whisper-large-v3-turbo"):
        """
        Инициализирует модель Whisper для транскрипции.
        """
        try:
            logger.info(f"Загрузка модели Whisper: {model_name}")
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info("Модель Whisper успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели Whisper: {e}")
            raise e

    def extract_audio_from_video(self, video_path: str, audio_path: str) -> None:
        """
        Извлекает аудио из видео файла и сохраняет его в формате WAV.
        """
        try:
            logger.info(f"Извлечение аудио из видео: {video_path} -> {audio_path}")
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Аудио успешно извлечено и сохранено в {audio_path}")
        except ffmpeg.Error as e:
            logger.error(f"Ошибка при извлечении аудио из видео: {e.stderr.decode()}")
            raise e

    def convert_audio_to_wav(self, input_path: str, output_path: str) -> None:
        """
        Конвертирует аудио файл в формат WAV с частотой дискретизации 16kHz и моно каналом.
        """
        try:
            logger.info(f"Конвертация аудио: {input_path} -> {output_path}")
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            logger.info(f"Аудио успешно конвертировано и сохранено в {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при конвертации аудио: {e}")
            raise e

    def transcribe(self, audio_path: str) -> str:
        """
        Транскрибирует аудио файл в текст.
        """
        try:
            logger.info(f"Транскрипция аудио файла: {audio_path}")
            # Загрузка аудио данных с использованием librosa
            audio, sampling_rate = librosa.load(audio_path, sr=16000)  # Whisper ожидает 16kHz
            # Обработка аудио данных
            inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            # Генерация транскрипции
            generated_ids = self.model.generate(input_features)
            # Декодирование транскрипции
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info("Транскрипция успешно завершена.")
            return transcription
        except Exception as e:
            logger.error(f"Ошибка при транскрипции аудио: {e}")
            raise e