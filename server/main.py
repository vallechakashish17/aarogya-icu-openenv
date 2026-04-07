from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Action(BaseModel):
    intervention: int

@app.get("/")
def home():
    return {"status": "Aarogya ICU API is Running"}

@app.post("/reset")
def reset():
    return {"hr": 120.0, "o2": 95.0, "tox": 0.0}

@app.post("/step")
def step(action: Action):
    # Standard OpenEnv response format
    return {"observation": [110.0, 96.0, 0.1], "reward": 0.5, "done": False}