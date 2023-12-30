import os
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt

file_directory = os.path.dirname(__file__)
csv_directory_train = os.path.join(file_directory, 'ROC_train.csv')
csv_directory_val = os.path.join(file_directory, 'ROC_validation.csv')
train_df = pd.read_csv(csv_directory_train)
val_df = pd.read_csv(csv_directory_val)

# Compute AUC's
train_auc = auc(train_df['False Positive Rate (x)'], train_df['True Positive Rate (y)'])
val_auc = auc(val_df['False Positive Rate (x)'], val_df['True Positive Rate (y)'])

# Plot ROC curve
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(train_df['False Positive Rate (x)'], train_df['True Positive Rate (y)'], label=f'Train (AUC = {train_auc:.3f})')
axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
axs[0].set_title('Train ROC Curve')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[1].plot(val_df['False Positive Rate (x)'], val_df['True Positive Rate (y)'], label=f'Validation (AUC = {val_auc:.3f})')
axs[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
axs[1].set_title('Validation ROC Curve')
axs[1].set_xlabel('False Positive Rate')
axs[1].set_ylabel('True Positive Rate')

# Add legend
axs[0].legend()
axs[1].legend()

plt.tight_layout()
plt.show()