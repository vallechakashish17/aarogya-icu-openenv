import os
import pandas as pd
import numpy as np
import torch as th
from openai import OpenAI
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom modules
from OpenEnvHealthcare import OpenEnvHealthcare
from data_pipeline import MedicalDataLoader
from masteraudit import MasterAudit 

def run_final_fix():
    # --- PHASE 0: PROXY INITIALIZATION ---
    # This is the FIX for the API error. 
    # Use os.environ["VAR"] to ensure it fails if the proxy isn't there.
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"]
    )

    # --- PHASE 1: AI MEDICAL TRAINING ---
    def make_env(): return OpenEnvHealthcare()
    env = DummyVecEnv([make_env])
    
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    print("\n--- Phase 1: Training AI Agent ---")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=30000)
    
    # --- PHASE 2: CLINICAL AUDIT ---
    loader = MedicalDataLoader("patients.csv")
    eval_env = OpenEnvHealthcare()
    auditor = MasterAudit() # Initialize your master audit logger
    results = [] 
    
    print("--- Phase 2: Running Clinical Audit ---")
    while True:
        p = loader.get_next_patient()
        if not p: break
        
        obs, _ = eval_env.reset(options={"custom_start": [p['hr'], p['o2'], p['tox']]})
        stable_count = 0
        
        for step_idx in range(30):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = eval_env.step(action)
            
            # Log to Master Audit
            auditor.log_step(p['id'], step_idx, obs, action, reward, info)
            
            if 60 <= obs[0] <= 100 and obs[1] >= 92:
                stable_count += 1
            if done or trunc: break
            
        score = round(stable_count / 30, 2)
        results.append({"id": p['id'], "score": score})

    # --- PHASE 3: LLM SUMMARY (The Proxy Call) ---
    avg_score = np.mean([r['score'] for r in results])
    
    print("--- Phase 3: Triggering LLM Proxy Call ---")
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{
            "role": "system", 
            "content": "You are a Chief Medical Officer auditing an AI agent."
        }, {
            "role": "user", 
            "content": f"The ICU AI achieved a stability score of {avg_score:.2f}. Summarize this."
        }]
    )
    
    cmo_feedback = response.choices[0].message.content
    print(f"CMO Feedback: {cmo_feedback}")

    # Finalize Audit and Save Files
    auditor.finalize(avg_score, cmo_feedback)
    pd.DataFrame(results).to_csv("final_clinical_report.csv", index=False)

if __name__ == "__main__":
    run_final_fix()