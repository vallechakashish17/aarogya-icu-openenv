from fastapi import FastAPI
from models import Observation, Action, Reward
from tasks import PatientEnv # Your existing environment logic

app = FastAPI()
env = PatientEnv()

@app.post("/reset")
def reset(task_id: str = "triage_easy"):
    obs = env.reset(task_id) # Returns [hr, o2, tox]
    return Observation(hr=obs[0], o2=obs[1], tox=obs[2], status="Initial")

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action.intervention)
    return Reward(value=reward, done=done, info=info)

@app.get("/state")
def get_state():
    return env.get_current_state()