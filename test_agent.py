import gymnasium as gym
from stable_baselines3 import PPO
import time

# SIMPLIFIED IMPORT: No 'envs.' prefix needed now
from healthcare_env import HealthcareEnv

# 1. Load the model
model = PPO.load("models/healthcare_agent_v1")

# 2. Run the test
env = HealthcareEnv()
obs, _ = env.reset()

print("--- STARTING PATIENT TRIAL ---")
for step in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    actions = ["Wait", "Lower HR", "Give O2"]
    print(f"Step {step+1}: AI chose {actions[action]} | HR: {obs[0]:.1f}, O2: {obs[1]:.1f}")
    
    if terminated:
        print("Trial Ended.")
        break
    time.sleep(0.5)

    import requests

# The Agent talks to the environment over the network
def ask_ai_doctor(action_name):
    response = requests.post("http://localhost:8000/step", json={"action": action_name})
    return response.json()

# Example: AI decides to give oxygen
result = ask_ai_doctor("oxygen")
print(f"New Vitals: {result['observation']}")