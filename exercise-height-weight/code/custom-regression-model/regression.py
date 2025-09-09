import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../../data/Cleaned.csv")
x = df[["Height"]]
y = df["Weight"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

prediction_heights = [[50], [100], [150], [200], [250]]
predictions = model.predict(prediction_heights)

for height, prediction in zip(prediction_heights, predictions):
    print(f"Predicted weight for {height[0]} cm: {prediction:.2f} kg")