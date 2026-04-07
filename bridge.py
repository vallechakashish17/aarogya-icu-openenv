from healthcare_env import HealthcareEnv

env = HealthcareEnv()

def doctor_tool(action_name):
    # Mapping text to the numbers the RL environment understands
    mapping = {"wait": 0, "medicate": 1, "oxygen": 2}
    action = mapping.get(action_name.lower(), 0)
    
    obs, reward, done, _, _ = env.step(action)
    
    return {
        "status": "CRITICAL" if done else "STABLE",
        "heart_rate": round(obs[0], 2),
        "oxygen_level": round(obs[1], 2),
        "reward_earned": reward
    }

# Example of an LLM-style call:
# print(doctor_tool("medicate"))