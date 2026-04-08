import uvicorn
import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# Move client creation inside a function to prevent boot-time crashes
def get_llm_client():
    return OpenAI(
        api_key=os.environ.get("API_KEY", "dummy_key"),
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    )

# ... your existing @app.post("/reset") and @app.post("/step") logic ...

def main():
    """The validator calls this function directly"""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()