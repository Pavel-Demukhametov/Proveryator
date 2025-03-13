import logging
import os
import numpy as np
from typing import Optional
from docx import Document
from PyPDF2 import PdfReader
import ffmpeg
from pydub import AudioSegment
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import websockets
from fastapi import FastAPI, WebSocket
import json
from starlette.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor
import asyncio
import whisperx
AudioSegment.converter = "D:\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста из PDF: {e}")
        raise e

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста из DOCX: {e}")
        raise e

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ошибка при извлечении текста из TXT: {e}")
        raise e

class AudioTranscription:
    batch_size=16
    compute_type = "float16"
    def __init__(self, model_name: str = "large-v2"):
        try:
            logger.info(f"Загрузка модели WhisperX: {model_name}")
            device = "cuda" 
            self.model = whisperx.load_model("large-v2", device, compute_type=self.compute_type)
            logger.info("Модель WhisperX успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели WhisperX: {e}")
            raise e

    def extract_audio_from_video(self, video_path: str, audio_path: str) -> None:
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
        try:
            logger.info(f"Конвертация аудио: {input_path} -> {output_path}")
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            logger.info(f"Аудио успешно конвертировано и сохранено в {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при конвертации аудио: {e}")
            raise e
    async def transcribe_audio_stream(self, websocket: WebSocket):
        await websocket.accept()
        
        # Получаем частоту дискретизации от клиента
        data = await websocket.receive_text()
        sample_rate = json.loads(data)['sampleRate']
        target_sr = 16000  # Целевая частота для Whisper
        accumulation_duration = 30  # Накапливаем 30 секунд аудио
        chunk_size = target_sr * accumulation_duration  # Размер чанка в сэмплах

        audio_buffer = np.array([], dtype=np.float32)
        temp_buffer = b''

        try:
            while True:
                # Получаем данные от клиента
                data = await websocket.receive_bytes()
                temp_buffer += data
                
                # Логируем размер temp_buffer
                logger.info(f"Получено байт: {len(data)}, Размер temp_buffer: {len(temp_buffer)}")

                # Проверяем, достаточно ли данных для преобразования
                if len(temp_buffer) >= 4:
                    valid_length = (len(temp_buffer) // 4) * 4
                    audio_chunk = np.frombuffer(temp_buffer[:valid_length], dtype=np.float32)
                    temp_buffer = temp_buffer[valid_length:]

                    # Приводим частоту, если клиент не отправляет 16 кГц
                    if sample_rate != target_sr:
                        audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=target_sr)

                    # Добавляем данные в audio_buffer
                    audio_buffer = np.concatenate((audio_buffer, audio_chunk))
                    logger.info(f"Размер audio_buffer: {len(audio_buffer)} сэмплов")

                    # Если набралось достаточно данных (30 секунд)
                    if len(audio_buffer) >= chunk_size:
                        logger.info(f"Накоплено {accumulation_duration} секунд аудио, передаем на транскрибацию")
                        chunk = audio_buffer[:chunk_size]
                        audio_buffer = audio_buffer[chunk_size:]  # Оставляем остаток, если есть

                        # Транскрибация
                        input_features = self.processor(chunk, sampling_rate=target_sr, return_tensors="pt").input_features.to(self.device)
                        generated_ids = self.model.generate(input_features, language='ru')
                        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Отправляем результат клиенту
                        await websocket.send_text(transcription)
                        logger.info(f"Отправлена транскрипция: {transcription}")

                        # Логируем оставшийся объем данных
                        logger.info(f"Остаток в audio_buffer: {len(audio_buffer)} сэмплов")

        except Exception as e:
            logger.error(f"Ошибка во время транскрипции: {e}")
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
    def transcribe(self, audio_path: str) -> str:
        try:
            logger.info(f"Транскрипция аудио файла: {audio_path}")
            # Загрузка и преобразование аудио
            audio = whisperx.load_audio(audio_path)
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            # Получение временных меток для слов
            result = whisperx.align(result["segments"], self.model, audio, self.device)
            # Объединение транскрипций
            full_transcription = " ".join([word["text"] for word in result["word_segments"]])
            logger.info("Транскрипция успешно завершена.")
            return full_transcription
        except Exception as e:
            logger.error(f"Ошибка при транскрипции аудио: {e}")
            raise e