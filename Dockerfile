FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the mandatory port for Hugging Face
EXPOSE 7860

# Run the FastAPI server using Uvicorn
# We bind to 0.0.0.0 and port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]