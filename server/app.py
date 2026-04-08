import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# SAFE VERSION: Prevents Internal Server Error if variables are missing
def get_llm_client():
    # Use .get() with a fallback to prevent "KeyError" or initialization crashes
    api_key = os.environ.get("API_KEY", "dummy_for_testing")
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    
    return OpenAI(api_key=api_key, base_url=base_url)

@app.post("/reset")
async def reset():
    try:
        # Initialize client safely
        client = get_llm_client()
        
        # YOUR RESET LOGIC HERE
        # Example: observation = my_env.reset()
        
        return {"observation": [140.0, 95.0, 0.0]} 
    except Exception as e:
        # This will show you exactly what is wrong in the Hugging Face logs
        print(f"Error during reset: {str(e)}")
        return {"error": "Internal server error occurred", "details": str(e)}

@app.get("/")
def health():
    return {"status": "Aarogya ICU API is Running"}