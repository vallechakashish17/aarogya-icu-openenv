from fastapi import FastAPI, HTTPException
from openenv_healthcare import OpenEnvHealthcare, MedicalAction, MedicalObservation
from pydantic import BaseModel
import uvicorn

# SCHEMA FOR API COMMUNICATION (Requirement 2)
class StepRequest(BaseModel):
    session_id: str
    action: MedicalAction

class StepResponse(BaseModel):
    session_id: str
    observation: MedicalObservation
    reward: float
    done: bool
    render: str

app = FastAPI(title="OpenEnv Healthcare API (Requirement 2)")
# In production you'd have a session manager
env = OpenEnvHealthcare() 

@app.post("/step", response_model=StepResponse)
async def step_agent(request: StepRequest):
    action_type = request.action.action_type
    
    # Map text action back to ID
    mapping = {"wait": 0, "administer_beta_blocker": 1, "administer_oxygen": 2}
    action_id = mapping[action_type]
    
    try:
        obs, reward, terminated, truncated, info = env.step(action_id)
        
        return StepResponse(
            session_id=request.session_id,
            observation=obs, # Already a Pydantic Model
            reward=reward,
            done=terminated or truncated,
            render=info["render"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    @app.get("/.well-known/mcp")
    def discover_tools():
        return {
        "mcp_version": "2026.1",
        "endpoints": [
            {
                "name": "administer_beta_blocker",
                "description": "Reduces heart rate. Required when BPM > 100. Side effect: Increases toxicity.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "administer_oxygen",
                "description": "Increases O2 saturation. Required when O2 < 95%.",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
    }