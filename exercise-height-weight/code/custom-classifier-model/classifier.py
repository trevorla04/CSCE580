import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../../data/Cleaned.csv")
x = df["Height"].astype(float).values
y, labels = pd.factorize(df["BMI Category"].astype(str))
x = x.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

prediction_heights = [[50], [100], [150], [200], [250]]
predictions = model.predict(prediction_heights)
prediction_labels = labels[predictions]

for height, prediction in zip(prediction_heights, predictions):
    print(f"Predicted class for {height[0]} cm: {prediction_labels[prediction]}")
