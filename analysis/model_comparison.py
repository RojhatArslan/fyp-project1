import pandas as pd
import matplotlib.pyplot as plt

results = {
    "Linear Regression": {
        "MAE": 0.54,
        "RMSE": 1.02,
        "R2": 0.86
    },
    "Random Forest": {
        "MAE": 0.04,
        "RMSE": 0.18,
        "R2": 0.996
    },
    "XGBoost": {
        "MAE": 0.08,
        "RMSE": 0.17,
        "R2": 0.996
    },
    "LightGBM": {
        "MAE": 0.10,
        "RMSE": 0.22,
        "R2": 0.993
    },
    "Prophet": {
        "MAE": 1.99,
        "RMSE": 2.98,
        "R2": -0.207
    }
}

results_df = pd.DataFrame(results).T
results_df.round(3)



results_df["RMSE"].sort_values().plot(kind="bar")
plt.ylabel("RMSE")
plt.title("Model Comparison Based on RMSE")
plt.tight_layout()
plt.show()

results_df["R2"].sort_values(ascending=False).plot(kind="bar")
plt.ylabel("R²")
plt.title("Model Comparison Based on R²")
plt.tight_layout()
plt.show()
results_df["MAE"].sort_values().plot(kind="bar")
plt.ylabel("MAE")
plt.title("Model Comparison Based on MAE")
plt.tight_layout()
plt.show()