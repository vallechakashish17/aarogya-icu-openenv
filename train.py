import gymnasium as gym
from stable_baselines3 import PPO
from healthcare_env import HealthcareEnv # Importing from your folder

# 1. Initialize Env
env = HealthcareEnv()

# 2. Setup Model (PPO is great for continuous/discrete health data)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

# 3. Train and Save
print("Starting medical training simulation...")
model.learn(total_timesteps=20000)
model.save("models/healthcare_agent_v1")
print("Training complete. Model saved in /models/")