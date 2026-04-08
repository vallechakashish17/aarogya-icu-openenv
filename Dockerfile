FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Serves the 'app' object from the 'app.py' file inside 'server' folder
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]