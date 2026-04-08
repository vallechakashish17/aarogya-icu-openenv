FROM python:3.11

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# CRITICAL: Point to the 'app' variable inside the 'app.py' file inside the 'server' folder
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]