import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def run_trial():
    # 1. Reset the patient via the API
    print("Resetting patient vitals...")
    response = requests.post(f"{BASE_URL}/reset")
    obs = response.json()["observation"] # This must match the server key
    for i in range(10):
        hr, o2 = obs
        print(f"Step {i+1} | HR: {hr:.1f}, O2: {o2:.1f}")

        # 2. Simple 'Agentic' Logic
        if hr > 100:
            action = 1  # Lower HR
            print("Action: Administering Beta-Blocker...")
        elif o2 < 92:
            action = 2  # Give Oxygen
            print("Action: Administering Oxygen...")
        else:
            action = 0  # Wait
            print("Action: Observing...")

        # 3. Send the action to the Server
        res = requests.post(f"{BASE_URL}/step", params={"action_id": action})
        data = res.json()
        obs = data["observation"]
        
        if data["done"]:
            print("Trial finished (Patient stabilized or critical).")
            break
        time.sleep(1)

if __name__ == "__main__":
    run_trial()