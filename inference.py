import os
import requests

# ─── Use the hackathon-injected proxy credentials ───────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]   # injected by the hackathon platform
API_KEY      = os.environ["API_KEY"]        # injected by the hackathon platform

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ─── Environment constants ────────────────────────────────────────────────────
MAX_STEPS   = 30
ACTIONS     = {0: "Wait", 1: "Beta-Blocker", 2: "Oxygen"}

# ─── Simple simulated ICU environment ────────────────────────────────────────
class ICUEnv:
    def __init__(self, task="triage_easy"):
        self.task = task
        self.reset()

    def reset(self):
        if self.task == "triage_easy":
            self.hr, self.o2, self.tox = 130, 95, 0.2
        elif self.task == "icu_balance_medium":
            self.hr, self.o2, self.tox = 125, 89, 0.5
        else:  # toxic_crisis_hard
            self.hr, self.o2, self.tox = 120, 91, 1.8
        self.step_count = 0
        return self._obs()

    def _obs(self):
        return {"hr": round(self.hr, 1), "o2": round(self.o2, 1), "tox": round(self.tox, 2)}

    def step(self, action):
        if action == 1:   # Beta-Blocker
            self.hr  = max(50, self.hr  - 8)
            self.tox = min(2.0, self.tox + 0.05)
        elif action == 2: # Oxygen
            self.o2  = min(100, self.o2 + 4)
            self.tox = min(2.0, self.tox + 0.02)
        else:             # Wait
            self.hr  = max(60, self.hr  - 1)
            self.o2  = min(100, self.o2 + 0.5)

        self.step_count += 1
        done = (
            self.step_count >= MAX_STEPS
            or self.tox >= 2.0
        )
        return self._obs(), done


# ─── Ask the LLM proxy which action to take ──────────────────────────────────
def choose_action(obs: dict, task: str) -> int:
    prompt = (
        f"You are an ICU decision-support AI.\n"
        f"Task: {task}\n"
        f"Current vitals — HR: {obs['hr']} bpm, O2: {obs['o2']}%, Toxicity: {obs['tox']}\n"
        f"Available actions:\n"
        f"  0 = Wait (do nothing)\n"
        f"  1 = Administer Beta-Blocker (lowers HR, raises toxicity slightly)\n"
        f"  2 = Administer Oxygen (raises O2, raises toxicity slightly)\n"
        f"Reply with ONLY a single digit: 0, 1, or 2."
    )

    payload = {
        "model": "gpt-4o-mini",   # LiteLLM proxy will route this
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5,
        "temperature": 0,
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()

    # Extract first digit found
    for ch in text:
        if ch in ("0", "1", "2"):
            return int(ch)
    return 0   # fallback: wait


# ─── Main agent loop ──────────────────────────────────────────────────────────
def run_baseline():
    print("[START]")

    for task in ["triage_easy", "icu_balance_medium", "toxic_crisis_hard"]:
        print(f"\n--- Task: {task} ---")
        env  = ICUEnv(task)
        obs  = env.reset()
        done = False

        while not done:
            action       = choose_action(obs, task)
            obs, done    = env.step(action)
            print(
                f"[STEP {env.step_count:02d}] HR={obs['hr']} O2={obs['o2']} "
                f"Tox={obs['tox']} -> Action: {action} ({ACTIONS[action]})"
            )

        print(f"[DONE] Final vitals: {obs}")

    print("\n[END]")


if __name__ == "__main__":
    run_baseline()