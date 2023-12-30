import os
import pandas as pd
import matplotlib.pyplot as plt

file_directory = os.path.dirname(__file__)
csv_directory = os.path.join(file_directory, 'epoch_metrics.csv')
df = pd.read_csv(csv_directory)

# Plot each metric
fig, axs = plt.subplots(2, 4, figsize=(12, 8))
metrics = ['Train MSE', 'Validation MSE', 'Train Accuracy', 'Validation Accuracy', 'Train Log Loss', 'Validation Log Loss', 'Train R²', 'Validation R²']
for i, metric in enumerate(metrics):
    axs[i%2, i//2].plot(df['Epoch'], df[metric])
    axs[i%2, i//2].set_title(metric)

plt.tight_layout()
plt.show()