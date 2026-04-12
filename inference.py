import os
import time
import requests

API_URL = os.getenv("API_BASE_URL", "https://Kash09-aarogya-icu-openenv.hf.space")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def run_baseline():
    print("[START]")

    try:
        r = requests.post(f"{API_URL}/reset", json={}, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        session_id = data["session_id"]
        print(f"[RESET] session_id={session_id}")
    except Exception as e:
        print(f"[RESET ERROR] {e}")
        return

    for step in range(30):
        try:
            state_r = requests.get(f"{API_URL}/state",
                                   params={"session_id": session_id},
                                   headers=HEADERS, timeout=30)
            state_r.raise_for_status()
            obs = state_r.json()["observation"]

            hr  = obs.get("heart_rate", 80)
            o2  = obs.get("o2_saturation", 98)

            if hr > 100:
                action_id = 1   # beta blocker
            elif o2 < 95:
                action_id = 2   # oxygen
            else:
                action_id = 0   # wait

            step_r = requests.post(f"{API_URL}/step",
                                   json={"session_id": session_id, "action_id": action_id},
                                   headers=HEADERS, timeout=30)
            step_r.raise_for_status()
            result = step_r.json()

            print(f"[STEP {step+1:02d}] HR:{hr} O2:{o2} -> action:{action_id} "
                  f"reward:{result['reward']:.2f} done:{result['done']}")

            if result["done"]:
                print("[DONE] Episode finished early.")
                break

        except Exception as e:
            print(f"[STEP {step+1} ERROR] {e}")
            time.sleep(1)

    print("[END]")

if __name__ == "__main__":
    run_baseline()