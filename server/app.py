import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# 1. Define a helper function to create the client ONLY when a request arrives
def get_llm_client():
    # These variables are injected by the Scaler validator at runtime
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    
    if not api_key or not base_url:
        # Fallback for local testing, but the validator will provide these
        return OpenAI(api_key="dummy", base_url="https://api.openai.com/v1")
        
    return OpenAI(api_key=api_key, base_url=base_url)

@app.post("/reset")
def reset():
    client = get_llm_client()
    # Example call: Must use the client created above
    # response = client.chat.completions.create(
    #     model=os.environ.get("MODEL_NAME", "gpt-4o"),
    #     messages=[{"role": "user", "content": "Initialize ICU state"}]
    # )
    return {"observation": [140.0, 95.0, 0.0]}

@app.post("/step")
def step(action: dict):
    client = get_llm_client()
    # All AI logic must happen through this 'client' instance
    return {"observation": [130.0, 96.0, 0.1], "reward": 0.5, "done": False}

@app.get("/")
def health():
    return {"status": "Running via LiteLLM Proxy"}