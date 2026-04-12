import os
import sys
import uuid
from openenv.core.env_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Models ──────────────────────────────────────────────────
class ICUAction(Action):
    action_id: int = Field(..., description="0=wait, 1=beta_blocker, 2=oxygen")

class ICUObservation(Observation):
    heart_rate: float = Field(..., description="Heart rate BPM")
    o2_saturation: float = Field(..., description="Oxygen saturation %")
    toxicity: float = Field(..., description="Toxicity level")
    step: int = Field(..., description="Current step")

# ── Environment ─────────────────────────────────────────────
class ICUEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.heart_rate = 120
        self.o2_saturation = 92
        self.toxicity = 0
        self.done = False

    def reset(self) -> ICUObservation:
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.heart_rate = 120
        self.o2_saturation = 92
        self.toxicity = 0
        self.done = False
        return self._obs()

    def _obs(self) -> ICUObservation:
        return ICUObservation(
            heart_rate=self.heart_rate,
            o2_saturation=self.o2_saturation,
            toxicity=self.toxicity,
            step=self._state.step_count,
        )

    def step(self, action: ICUAction):
        reward = 0.0
        if action.action_id == 1:
            self.heart_rate = max(60, self.heart_rate - 10)
            self.toxicity += 5
            reward = 1.0 if self.heart_rate <= 100 else 0.5
        elif action.action_id == 2:
            self.o2_saturation = min(100, self.o2_saturation + 3)
            reward = 1.0 if self.o2_saturation >= 95 else 0.5
        else:
            reward = -0.1

        self._state.step_count += 1
        self.done = self._state.step_count >= 30 or self.toxicity >= 50
        return self._obs(), reward, self.done

# ── Create app using OpenEnv framework ──────────────────────
app = create_app(ICUEnvironment, ICUAction, ICUObservation, env_name="aarogya-icu")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
