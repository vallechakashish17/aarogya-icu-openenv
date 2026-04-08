import os
from openai import OpenAI

# Use a default string like "EMPTY" so the client doesn't crash on boot
# The validator will replace these with real values when it runs its tests
client = OpenAI(
    api_key=os.environ.get("API_KEY", "EMPTY_KEY_FOR_BUILD"),
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
)

@app.get("/")
def read_root():
    return {"status": "Aarogya ICU API is Running"}