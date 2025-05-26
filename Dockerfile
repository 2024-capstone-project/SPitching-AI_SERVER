FROM python:3.10.12-slim
WORKDIR /AI_SERVER

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /AI_SERVER/requirements.txt
RUN pip install -r /AI_SERVER/requirements.txt

COPY ./label /AI_SERVER/label
COPY ./models /AI_SERVER/models
COPY ./app /AI_SERVER/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]