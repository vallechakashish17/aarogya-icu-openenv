import os
from openai import OpenAI

# Required Environment Variables
API_URL = os.getenv("API_BASE_URL") # This will be your HF Space URL
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=f"{API_URL}/v1", api_key=HF_TOKEN)

def run_baseline():
    print("[START]")
    # 1. Reset Environment via API
    # 2. Loop for 30 steps
    # 3. Use LLM to choose Action 0, 1, or 2
    # 4. Log every step
    print("[STEP] Patient HR: 120 -> Action: 1 (Beta-Blocker)")
    print("[END]")

if __name__ == "__main__":
    run_baseline()