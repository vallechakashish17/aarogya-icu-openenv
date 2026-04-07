import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs import HealthcareEnv

env = HealthcareEnv()
model = PPO.load("models/healthcare_agent_v1")

# Store data for plotting
hr_history = []
o2_history = []
rewards = []

obs, _ = env.reset()
for _ in range(30): # Run for 30 "minutes"
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    
    hr_history.append(obs[0])
    o2_history.append(obs[1])
    rewards.append(reward)
    if done: break

# Plotting the Vitals
plt.figure(figsize=(10, 5))
plt.plot(hr_history, label="Heart Rate (BPM)", color='red', marker='o')
plt.plot(o2_history, label="Oxygen Saturation (%)", color='blue', marker='x')
plt.axhline(y=90, color='gray', linestyle='--', label="Safe HR Upper Bound")
plt.axhline(y=95, color='green', linestyle='--', label="Safe O2 Lower Bound")
plt.title("Patient Vital Signs under AI Management")
plt.xlabel("Time Steps")
plt.legend()
plt.show()