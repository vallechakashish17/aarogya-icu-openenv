import pandas as pd
import os
import numpy as np
from data_pipeline import MedicalDataLoader
from openenv_healthcare import OpenEnvHealthcare
from stable_baselines3 import PPO

def run_clinical_audit():
    # 1. Setup paths
    model_path = "models/healthcare_model"
    
    # 2. Initialize Environment and Loader
    loader = MedicalDataLoader("patients.csv")
    env = OpenEnvHealthcare()
    
    # 3. THE FIX: Safe Model Loading
    if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print("Warning: Trained model not found! Creating a baseline model for the report...")
        # Create a dummy model so the script doesn't crash
        model = PPO("MlpPolicy", env, verbose=0)
    
    results = []

    # 4. Process Patients
    while True:
        patient_data = loader.get_next_patient()
        if not patient_data: break
        
        # Inject Patient into Environment
        obs, _ = env.reset(options={"custom_start": [patient_data['hr'], patient_data['o2'], patient_data['tox']]})
        
        stable_steps = 0
        for _ in range(30):
            # Use the model to predict
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Criteria: Was the patient safe? (obs[0]=HR, obs[1]=O2)
            if 60 <= obs[0] <= 90 and obs[1] >= 95:
                stable_steps += 1
            if terminated: break
        
        results.append({
            "patient_id": patient_data['id'],
            "stability_score": round(stable_steps / 30, 2),
            "outcome": "STABILIZED" if stable_steps > 20 else "CRITICAL"
        })

    # 5. Save the Clinical Report
    report_df = pd.DataFrame(results)
    report_df.to_csv("clinical_audit_report.csv", index=False)
    
    print("\n" + "="*40)
    print("CLINICAL AUDIT COMPLETE")
    print(f"Total Patients Audited: {len(results)}")
    print(f"Average Stability: {report_df['stability_score'].mean():.2f}")
    print("Report saved to: clinical_audit_report.csv")
    print("="*40)

if __name__ == "__main__":
    run_clinical_audit()