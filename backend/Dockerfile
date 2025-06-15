FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app


RUN apt-get update && \
    apt-get install -y libsndfile1 && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
COPY wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt

COPY . .

RUN python download_models.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



