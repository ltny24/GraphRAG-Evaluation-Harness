"""
Milestone 3 - Evaluation Harness
Step 3.3: Plot Results

Reads the CSV from the evaluator and generates comparison charts.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics():
    print("Plotting results...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "eval"
    csv_file = data_dir / "results.csv"
    
    if not csv_file.exists():
        print(f"Error: {csv_file} not found. Run evaluator.py first.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Calculate averages
    avg_scores = df.groupby('system')[['correctness', 'faithfulness']].mean().reset_index()
    
    # Order systems logically
    system_order = ['Vector', 'Graph', 'Hybrid']
    avg_scores['system'] = pd.Categorical(avg_scores['system'], categories=system_order, ordered=True)
    avg_scores = avg_scores.sort_values('system')
    
    # Plot Correctness
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_scores['system'], avg_scores['correctness'], color=['#3498db', '#e74c3c', '#2ecc71'])
    
    plt.title('Correctness Score Comparison (Vector vs Graph vs Hybrid RAG)', fontsize=14, pad=20)
    plt.ylabel('Correctness Score (0-100)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_img = data_dir / "correctness_chart.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {output_img}")

if __name__ == "__main__":
    plot_metrics()
