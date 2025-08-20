import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
csv_path = "/home/dandi/smt8/paperstegano/results/chi_square/results_chisquare_detection.csv"  # Update this to your actual path if needed
df = pd.read_csv(csv_path)

# Prepare data for plotting
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="rate", y="p_value", hue="channel")

# Styling
plt.title("Boxplot p-value Hasil Uji Chi-Square untuk Berbagai Embedding Rate")
plt.xlabel("Embedding Rate (%)")
plt.ylabel("p-value")
plt.legend(title="Channel")
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
