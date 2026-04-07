import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from openenv_healthcare import OpenEnvHealthcare
from data_pipeline import MedicalDataLoader

def run_final_fix():
    # 1. Setup Environment
    def make_env(): return OpenEnvHealthcare()
    env = DummyVecEnv([make_env])
    
    # 2. Train the AI
    print("\n--- Phase 1: AI Medical Training (Residency) ---")
    model = PPO("MlpPolicy", env, verbose=1,learning_rate=0.0001)
    model.learn(total_timesteps=30000)
    
    # 3. Audit Patients from CSV
    loader = MedicalDataLoader("patients.csv")
    eval_env = OpenEnvHealthcare()
    results = [] 
    
    print("--- Phase 2: Running Clinical Audit on CSV Data ---")
    while True:
        p = loader.get_next_patient()
        if not p: break
        
        obs, _ = eval_env.reset(options={"custom_start": [p['hr'], p['o2'], p['tox']]})
        stable_count = 0
        
        for _ in range(30):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = eval_env.step(action)
            # Check if patient is in the "Safe Zone"
            if 60 <= obs[0] <= 100 and obs[1] >= 92:
                stable_count += 1
            if done or trunc: break
            
        results.append({
            "id": p['id'], 
            "score": round(stable_count / 30, 2)
        })

    # 4. Save the CSV
    pd.DataFrame(results).to_csv("final_clinical_report.csv", index=False)

    # 5. THE TERMINAL TABLE (The part you want to see!)
    print("\n" + "="*45)
    print(f"{'PATIENT ID':<15} | {'STABILITY SCORE':<15} | {'STATUS'}")
    print("-" * 45)
    
    for r in results:
        # Visual indicator for the user
        status = "✅ STABILIZED" if r['score'] > 0.80 else "⚠️ CRITICAL"
        print(f"{r['id']:<15} | {r['score']:<15.2f} | {status}")
    
    print("-" * 45)
    avg_score = np.mean([r['score'] for r in results])
    print(f"OVERALL POPULATION STABILITY: {avg_score:.2f}")
    print("="*45)
    print("Report saved to: final_clinical_report.csv\n")

if __name__ == "__main__":
    run_final_fix()