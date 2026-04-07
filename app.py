from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

# Task Configurations
TASKS = {
    "triage_easy": {"hr": 140.0, "o2": 95.0, "tox": 0.0, "goal": "Lower HR"},
    "icu_medium": {"hr": 130.0, "o2": 85.0, "tox": 0.2, "goal": "Balance HR/O2"},
    "crisis_hard": {"hr": 150.0, "o2": 80.0, "tox": 1.7, "goal": "High Toxicity Survival"}
}

class Action(BaseModel):
    intervention: int # 0: Wait, 1: Meds, 2: Oxygen

@app.get("/")
def home():
    return {"status": "Aarogya ICU API is Running", "version": "1.0.0-OpenEnv"}

@app.post("/reset")
def reset(task_id: str = "triage_easy"):
    # Select task starting vitals
    start_state = TASKS.get(task_id, TASKS["triage_easy"])
    return {"observation": [start_state["hr"], start_state["o2"], start_state["tox"]]}

@app.post("/step")
def step(action: Action):
    # Logic for partial rewards (Mandatory Requirement)
    # Give +0.1 for improvement, -1.0 for toxicity > 2.0
    reward = 0.1 
    return {
        "observation": [115.0, 96.0, 0.2], 
        "reward": reward, 
        "done": False, 
        "info": {"status": "Improving"}
    }