FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Azure App Service expects the app to listen on PORT env variable
ENV PORT=8080

CMD gunicorn --bind 0.0.0.0:$PORT app:app