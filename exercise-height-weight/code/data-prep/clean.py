import pandas as pd
import numpy as np

df = pd.read_csv('../../data/DataSample-WeightHeight.csv')

# C1 - underweight <18.5
# C2 - Healthy 18.5 - 25
# C3 - Overweight 25 - 30
# C4 - Obese > 30

df = df.drop(df.index[0])
print(df.head())
print(df.info())

df["Height"] = pd.to_numeric(df["Unnamed: 1"], errors="coerce")
df["Weight"] = pd.to_numeric(df["Unnamed: 2"], errors="coerce")
df = df.dropna(subset=["Height", "Weight"]).reset_index(drop=True)

X = df[["Height"]].to_numpy(dtype=float)
y = df["Weight"].to_numpy(dtype=float)

X_b = np.c_[np.ones((X.shape[0], 1)), X]

theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

intercept, slope = theta
print ("Intercept: ", intercept)
print ("Slope: ", slope)

height_new = np.array([[1, 50]]) # [1, height] for reshaped
weight_pred = height_new @ theta
print("Predicted weight for 50 cm:", weight_pred[0])

def category(bmi):
    if (bmi < 18.5):
        return "C1: Underweight"
    if (bmi < 25):
        return "C2: Healthy"
    if (bmi < 30):
        return "C3: Overweight"
    else:
        return "C4: Obese"

df["Unnamed: 3"] = pd.to_numeric(df["Unnamed: 3"], errors="coerce")
df["Unnamed: 4"] = df["Unnamed: 3"].apply(category)
print(df.head())



