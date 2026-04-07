import pandas as pd
import matplotlib.pyplot as plt

def generate_comparison_plot():
    # 1. Load the population-level data
    # 'dumb_avg' is your old score (0.25). 'smart_avg' is your new score (0.82).
    data = {
        'Scenario': ['Random Baseline', 'Trained Doctor AI'],
        'Average Stability Score': [0.25, 0.82]
    }
    df = pd.DataFrame(data)

    # 2. Setup the plot area
    plt.figure(figsize=(9, 6))
    plt.title('Patient Stabilization Performance: OpenEnv Healthcare Simulation', fontsize=16)
    plt.ylabel('Stability Score (higher is better)', fontsize=12)
    plt.ylim(0.0, 1.0) # Set Y-axis scale from 0 to 1

    # 3. Create the bar chart
    bars = plt.bar(df['Scenario'], df['Average Stability Score'], color=['#e74c3c', '#2ecc71'])

    # 4. Add the score labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 5. Add Grid and Save
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("clinical_performance_comparison.png", dpi=150)
    print("Chart successfully generated: clinical_performance_comparison.png")

if __name__ == "__main__":
    try:
        generate_comparison_plot()
    except ImportError as e:
        print(f"Error: Missing libraries. Please run: pip install pandas matplotlib")