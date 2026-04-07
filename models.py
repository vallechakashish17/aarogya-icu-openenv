from pydantic import BaseModel
from typing import List, Optional

class Observation(BaseModel):
    hr: float
    o2: float
    tox: float
    patient_status: str

class Action(BaseModel):
    intervention: int # 0: Wait, 1: Beta-Blocker, 2: Oxygen

class Reward(BaseModel):
    value: float
    done: bool
    info: dict