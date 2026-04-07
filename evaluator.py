import numpy as np
from openenv_healthcare import OpenEnvHealthcare
from stable_baselines3 import PPO

def grade_agent(task_id):
    # 1. Start the environment
    env = OpenEnvHealthcare(task_id=task_id)
    
    # 2. DEFINE THE MODEL HERE (This stops the NameError)
    try:
        from stable_baselines3 import PPO
        # Make sure 'models/healthcare_model' is the correct path to your file
        model = PPO.load("models/healthcare_model")
    except Exception as e:
        # Fallback if the file is missing so the code doesn't crash
        model = None
        print(f"Warning: Could not load model for {task_id}. Error: {e}")

    obs, _ = env.reset()
    stable_steps = 0
    
    for _ in range(30):
        # 3. Use the model variable we just defined
        if model is not None:
            # Note: ensure obs is a numpy array for PPO
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample() # Random action if model fails
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check stability criteria
        if 60 <= obs[0] <= 90 and obs[1] >= 95:
            stable_steps += 1
            
        if terminated or truncated:
            break
            
    return stable_steps / 30.0