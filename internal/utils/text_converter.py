import logging
import os
from typing import Optional
from docx import Document
from PyPDF2 import PdfReader
import ffmpeg
from pydub import AudioSegment
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

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
    def __init__(self, model_name: str = "openai/whisper-large-v3-turbo", top_p: float = 0.9, top_k: int = 50):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.top_p = top_p
        self.top_k = top_k



    def extract_audio_from_video(self, video_path: str, audio_path: str) -> None:
        try:
            logger.info(f"Извлечение аудио из видео: {video_path} -> {audio_path}")
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=False)
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

    def transcribe(self, audio_path: str) -> str:

        logger.info(f"Транскрипция аудио файла: {audio_path}")
        audio, sampling_rate = librosa.load(audio_path, sr=16000)
        chunk_length_s = 30
        stride_length_s = 5
        chunk_length = chunk_length_s * sampling_rate
        stride_length = stride_length_s * sampling_rate
        total_length = len(audio)
        chunks = []
        start = 0
        while start < total_length:
            end = start + chunk_length
            chunk = audio[start:end]
            chunks.append(chunk)
            start += chunk_length - stride_length
        logger.info(f"Разделено на {len(chunks)} чанков по {chunk_length_s} секунд.")
        transcriptions = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"Транскрипция чанка {idx + 1}/{len(chunks)}")
            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt", language='ru', task="transcribe")
            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device) if 'attention_mask' in inputs else None
            generated_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=self.top_p,
                top_k=self.top_k
            )

            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcriptions.append(transcription)

        full_transcription = " ".join(transcriptions)
        return full_transcription