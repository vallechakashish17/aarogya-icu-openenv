import os
from stable_baselines3 import PPO
from openenv_healthcare import OpenEnvHealthcare

def train_doctor_ai():
    # 1. Create the ICU Environment
    env = OpenEnvHealthcare()
    
    # 2. Ensure the 'models' folder exists
    if not os.path.exists("models"):
        os.makedirs("models")

    # 3. Define the Agent (PPO is the standard 'Doctor' algorithm)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    print("--- AI is now entering Medical Residency (Training) ---")
    
    # 4. Let the AI practice for 10,000 steps
    model.learn(total_timesteps=50000)

    # 5. Save the 'Smart' version of the AI
    model.save("models/healthcare_model")
    print("\nSUCCESS: Smart model saved to models/healthcare_model.zip")

if __name__ == "__main__":
    train_doctor_ai()