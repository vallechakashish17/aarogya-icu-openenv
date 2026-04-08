import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# 1. Define a helper to get the client only when needed
def get_llm_client():
    api_key = os.environ.get("API_KEY", "dummy_key_for_boot")
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)

# 2. Update your endpoints to use the helper
@app.post("/reset")
def reset():
    client = get_llm_client() # Client is created here, NOT at the top of the file
    # ... your reset logic using client ...
    return {"observation": [140.0, 95.0, 0.0]}

@app.post("/step")
def step(action: dict):
    client = get_llm_client() # Client is created here safely
    # ... your step logic using client ...
    return {"observation": [130.0, 96.0, 0.1], "reward": 0.5, "done": False}

@app.get("/")
def health():
    return {"status": "Aarogya ICU API is Running"}

# ... rest of your code (main function, etc.) ...