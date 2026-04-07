import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from openenv_healthcare import OpenEnvHealthcare

def grade_agent(task_id):
    # 1. Initialize the environment
    env = OpenEnvHealthcare(task_id=task_id)
    
    # 2. LOAD THE MODEL (The fix for your NameError)
    try:
        # Replace 'models/healthcare_model' with your actual filename
        model = PPO.load("models/healthcare_model")
        print(f"Loaded trained model for {task_id}")
    except:
        model = None
        print(f"Warning: No model found for {task_id}. Using random actions.")

    # 3. Reset and start the loop
    obs, _ = env.reset()
    stable_steps = 0
    total_steps = 30

    for _ in range(total_steps):
        # Extract vitals for the prediction
        sb3_obs = np.array([obs[0], obs[1], obs[2]], dtype=np.float32)

        # 4. Use the model if it exists, otherwise sample randomly
        if model is not None:
            action, _ = model.predict(sb3_obs, deterministic=True)
        else:
            action = env.action_space.sample()

        # 5. Take the step
        obs, reward, terminated, truncated, info = env.step(action)

        # Check for stability (HR: 60-90, O2: >95)
        if 60 <= obs[0] <= 90 and obs[1] >= 95:
            stable_steps += 1

        if terminated or truncated:
            break

    return stable_steps / total_steps

if __name__ == "__main__":
    tasks = ["tachycardia_simple", "hypoxia_medium", "critical_crash"]
    scores = {t: grade_agent(t) for t in tasks}
    
    print("\n" + "="*30)
    for t, s in scores.items():
        print(f"Task: {t} | Score: {s:.2f}")
    print(f"FINAL OPENENV GRADE: {np.mean(list(scores.values())):.2f}")
    print("="*30)