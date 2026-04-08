import os
import uvicorn
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# Phase 2: Helper to use the LiteLLM Proxy
def get_llm_client():
    return OpenAI(
        api_key=os.environ.get("API_KEY", "dummy_key"),
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    )

@app.get("/")
def health():
    return {"status": "Aarogya ICU API is Running"}

@app.post("/reset")
def reset():
    client = get_llm_client()
    # Your reset logic here...
    return {"observation": [140.0, 95.0, 0.0]}

@app.post("/step")
def step(action: dict):
    client = get_llm_client()
    # Your step logic here...
    return {"observation": [130.0, 96.0, 0.1], "reward": 0.5, "done": False}

# Phase 1: The 'main' function and 'callable' block
def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()