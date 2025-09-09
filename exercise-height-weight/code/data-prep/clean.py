import pandas as pd

original = pd.read_csv('../../data/PreCleaning.csv')
df = original.copy()

df = df.drop(df.index[0])

df["Height"] = pd.to_numeric(df["Unnamed: 1"], errors="coerce")
df["Weight"] = pd.to_numeric(df["Unnamed: 2"], errors="coerce")
df = df.dropna(subset=["Height", "Weight"]).reset_index(drop=True)

df["BMI"] = (df["Weight"] / ((df["Height"]/100) ** 2))

def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "C1"
    elif bmi < 25:
        return "C2"
    elif bmi < 30:
        return "C3"
    else:
        return "C4"
df["BMI Category"] = df["BMI"].apply(bmi_category)

df.to_csv("../../data/Cleaned.csv", index=False)
