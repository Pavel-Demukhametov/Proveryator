FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Устанавливаем ffmpeg и сертификаты
RUN apt-get update && apt-get install -y \
    ffmpeg \
    ca-certificates \
    && update-ca-certificates \
    && apt-get clean

WORKDIR /fast

COPY requirements.txt ./
RUN pip install --default-timeout=640 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]