FROM python:3.11-slim

WORKDIR /app

# System deps some scientific/ML packages need to build wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces (Docker SDK) requires the app to listen on port 7860
ENV MODEL_DIR=/tmp
EXPOSE 7860

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
