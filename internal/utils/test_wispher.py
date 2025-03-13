import io
import json
import logging
import sys
import time
from datetime import datetime
from faster_whisper import WhisperModel
from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('faster-whisper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

device = "cuda"
model_path = r"D:\Program Files\Lecture_test_front\fast\internal\utils\faster-whisper-large-v3-turbo-russian"
model = WhisperModel(model_path, device)

audio = AudioSegment.from_file("50.mp4", format="mp4")

chunk_length = 30 * 1000
chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

logging.info(f'Start transcribe at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
start = time.time()

transcribed_text = []
for i, chunk in enumerate(chunks):
    buffer = io.BytesIO()
    chunk.export(buffer, format="wav")
    buffer.seek(0)
    segments, info = model.transcribe(buffer, language="ru")
    chunk_text = "".join(segment.text for segment in segments)
    transcribed_text.append(chunk_text)
    logging.info(f"Chunk {i+1}/{len(chunks)} transcribed: {chunk_text}")

end = time.time()
logging.info(f'Finish transcribe at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
logging.info(f'Total time: {end - start:.2f} seconds')
logging.info(f'Transcribed Text: {transcribed_text}')
