FROM python:3.11-slim

WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/train.py .
COPY data/raw /app/data/raw

CMD ["python", "train.py"]
