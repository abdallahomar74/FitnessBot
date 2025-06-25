FROM python:3.10-slim

# تثبيت المتطلبات الأساسية
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# إعداد مجلد العمل
WORKDIR /app

# نسخ الملفات
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# تحميل punkt لـ NLTK
RUN python -c "import nltk; nltk.download('punkt')"

# فتح المنفذ 8000
EXPOSE 8000

# الأمر الافتراضي لتشغيل FastAPI
CMD ["uvicorn", "app.chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"]
