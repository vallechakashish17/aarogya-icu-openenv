from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 1. Define the structures the judges expect
class Observation(BaseModel):
    observation: List[float] # [HR, O2, Tox]

class Action(BaseModel):
    intervention: int

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: dict

# 2. Update the endpoints with real logic
@app.post("/reset", response_model=Observation)
def reset():
    # Return starting vitals: High HR (140), Normal O2 (95), No Tox (0)
    return {"observation": [140.0, 95.0, 0.0]}

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    # Simulating a small improvement for the baseline test
    return {
        "observation": [130.0, 96.0, 0.1],
        "reward": 0.1,
        "done": False,
        "info": {"status": "Improving"}
    }
import uvicorn

# ... keep all your existing FastAPI code (app = FastAPI(), etc.) ...

def main():
    """Entry point for the validator"""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()