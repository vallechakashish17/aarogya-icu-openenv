print("SERVER STARTED LOADING")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

# ── inline environment ──────────────────────────────────────
class ICUEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.heart_rate = 120
        self.o2_saturation = 92
        self.toxicity = 0
        self.done = False
        return self._obs()

    def _obs(self):
        return {
            "heart_rate": self.heart_rate,
            "o2_saturation": self.o2_saturation,
            "toxicity": self.toxicity,
            "step": self.step_count,
        }

    def step(self, action_id: int):
        reward = 0.0
        if action_id == 1:  # beta blocker
            self.heart_rate = max(60, self.heart_rate - 10)
            self.toxicity += 5
            reward = 1.0 if self.heart_rate <= 100 else 0.5
        elif action_id == 2:  # oxygen
            self.o2_saturation = min(100, self.o2_saturation + 3)
            reward = 1.0 if self.o2_saturation >= 95 else 0.5
        else:  # wait
            reward = -0.1

        self.step_count += 1
        self.done = self.step_count >= 30 or self.toxicity >= 50
        return self._obs(), reward, self.done

# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(title="Aarogya ICU OpenEnv")

sessions: dict = {}  # session_id -> ICUEnvironment

# ── Models ──────────────────────────────────────────────────
class ResetRequest(BaseModel):
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    session_id: str
    action_id: int  # 0=wait, 1=beta_blocker, 2=oxygen

# ── Inference logic (mirrors inference.py) ──────────────────
def get_action(obs: dict) -> int:
    hr = obs.get("heart_rate", 80)
    o2 = obs.get("o2_saturation", 98)
    if hr > 100:
        return 1   # beta blocker
    elif o2 < 95:
        return 2   # oxygen
    else:
        return 0   # wait

# ── Endpoints ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Aarogya ICU API is Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest = None):
    sid = (req.session_id if req and req.session_id else None) or str(uuid.uuid4())
    env = ICUEnvironment()
    sessions[sid] = env
    obs = env.reset()
    return {"session_id": sid, "observation": obs}

@app.get("/state")
def state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "observation": sessions[session_id]._obs()}

@app.post("/step")
def step(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    env = sessions[req.session_id]
    if env.done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")
    obs, reward, done = env.step(req.action_id)
    return {
        "session_id": req.session_id,
        "observation": obs,
        "reward": reward,
        "done": done,
        "render": f"Step {obs['step']} | HR:{obs['heart_rate']} O2:{obs['o2_saturation']} Tox:{obs['toxicity']}",
    }

@app.post("/predict")
def predict(req: ResetRequest = None):
    """Auto-infer action from current state of a session."""
    sid = req.session_id if req and req.session_id else None
    if not sid or sid not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    obs = sessions[sid]._obs()
    action = get_action(obs)
    return {"session_id": sid, "observation": obs, "recommended_action": action}

@app.get("/.well-known/mcp")
def discover_tools():
    return {
        "mcp_version": "2026.1",
        "endpoints": [
            {
                "name": "administer_beta_blocker",
                "description": "Reduces heart rate. Required when BPM > 100.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "administer_oxygen",
                "description": "Increases O2 saturation. Required when O2 < 95%.",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)