import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_patient_comparison():
    target_file = "final_clinical_report.csv"
    
    # 1. Check if the file actually exists before trying to read it
    if not os.path.exists(target_file):
        print(f"❌ ERROR: '{target_file}' not found in {os.getcwd()}")
        print("👉 FIX: You must run 'python master_audit.py' first!")
        return

    # 2. Load the data
    df = pd.read_csv(target_file)
    print(f"✅ Successfully loaded {len(df)} patients from {target_file}")

    # 3. Create the Chart
    plt.figure(figsize=(10, 6))
    # Color logic: Green for good, Yellow for okay, Red for bad
    colors = ['#2ecc71' if x > 0.8 else '#f1c40f' if x > 0.6 else '#e74c3c' for x in df['stability_score']]
    
    bars = plt.bar(df['patient_id'], df['stability_score'], color=colors)
    
    plt.title('Individual Patient Stability Comparison', fontsize=14)
    plt.ylabel('Stability Score (0.0 - 1.0)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center')

    plt.savefig("patient_comparison_details.png")
    print("📈 Chart saved as 'patient_comparison_details.png'")

if __name__ == "__main__":
    plot_patient_comparison()