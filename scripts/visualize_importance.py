import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 1. Load model and features
clf = joblib.load('data/activity_model.pkl')
# These names must match the ones in the updated train_activity_classifier.py
feature_names = [
    'acc_std', 'acc_mean', 'acc_max', 
    'gyr_std', 'gyr_mean', 'gyr_max', 
    'jerk_std', 'jerk_max', 
    'roll_std', 'pitch_std', 'yaw_std', 
    'pitch_mean'
]

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

# 2. Plot
plt.figure(figsize=(10, 6))
forest_importances.sort_values(ascending=False).plot.bar(yerr=std, color='teal')
plt.title("Importancia de las Características (Feature Importance)")
plt.ylabel("Disminución media de la impureza")
plt.tight_layout()

output_path = 'docs/plots/feature_importance.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Feature importance plot saved to {output_path}")
