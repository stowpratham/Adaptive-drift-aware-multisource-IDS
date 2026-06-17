import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("results/baselines", exist_ok=True)

# Load comparison data
df = pd.read_csv("results/baselines/final_comparison.csv")

print("Generating comparison plots...")

metrics = {
    "Accuracy": "accuracy_comparison.png",
    "F1": "f1_comparison.png",
    "FAR": "far_comparison.png",
    "DetectionRate": "detection_rate_comparison.png",
}

for metric, filename in metrics.items():

    plt.figure(figsize=(8, 5))

    bars = plt.bar(df["Model"], df[metric])

    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xlabel("Models")

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    save_path = os.path.join(
        "results",
        "baselines",
        filename,
    )

    plt.savefig(save_path)

    print(f"✓ Saved: {save_path}")

    plt.close()

print("\nAll comparison plots generated successfully.")