FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV FF1_CACHE_DIR=/tmp/cache
ENV MODEL_DIR=/tmp

CMD ["uvicorn", "app.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860"]
