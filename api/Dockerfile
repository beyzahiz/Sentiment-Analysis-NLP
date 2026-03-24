FROM python:3.10

WORKDIR /app

# Gerekli sistem paketleri (bazı kütüphaneler için gerekebilir)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ÖNEMLİ: Uvicorn'u direkt başlatmak yerine, 
# Render'ın portu daha kolay yakalaması için şu formatı kullanıyoruz:
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]