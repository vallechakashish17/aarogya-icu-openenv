import os
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def health():
    return {"status": "Aarogya ICU API is Running"}

@app.post("/reset")
def reset():
    # Logic for patient initialization
    return {"observation": [140.0, 95.0, 0.0]}

@app.post("/step")
def step(action: dict):
    # Logic for medical intervention
    return {"observation": [130.0, 96.0, 0.1], "reward": 0.5, "done": False}

def main():
    # The port MUST be 7860 for Hugging Face
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()