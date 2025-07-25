import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Title of the app
st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower's measurements and let the model predict the Iris species!")

# Input sliders for user to enter measurements
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.2)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Predict the class
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
predicted_class = iris.target_names[prediction][0]

# Display the prediction
st.success(f"🌼 Predicted Iris Species: *{predicted_class}*")

# Optional: Show the dataset (if user checks box)
if st.checkbox("Show raw Iris dataset"):
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["Species"] = pd.Series(y).map(dict(enumerate(iris.target_names)))
    st.dataframe(df)
