# 🌦️ KNN Weather Classifier

An interactive machine learning web application that uses the K-Nearest Neighbors (KNN) algorithm to classify weather conditions based on environmental features.

## 🚀 Overview
This project demonstrates a supervised learning classification model. Users can manipulate environmental variables to see how a KNN model draws boundaries between "Sunny" and "Rainy" weather conditions.

## 🛠️ Tech Stack
* **Framework:** Streamlit (UI/UX)
* **Machine Learning:** Scikit-Learn (KNeighborsClassifier)
* **Data Processing:** NumPy
* **Visualization:** Matplotlib

## 💡 Key Features
* **Live Prediction:** Real-time classification updates as you move the temperature and humidity sliders.
* **Dynamic K-Value:** Adjust the number of neighbors (K) to see how it affects model confidence and classification.
* **Visual Decision Plot:** An interactive scatter plot showing the training data points and where the new input sits in the feature space.
* **Probability Metrics:** Displays the mathematical confidence (probability) for both Sunny and Rainy labels.

## 📖 How the Model Works
The model utilizes a small training set of temperature and humidity pairings:
1. **Distance Calculation:** Measures the Euclidean distance between the user input and stored training points.
2. **Neighbor Selection:** Identifies the $K$ closest points.
3. **Voting:** Assigns the label based on the majority class among the $K$ neighbors.

> **Model Logic:** Generally, higher temperatures and lower humidity lead to a "Sunny" prediction, while lower temperatures and higher humidity trend toward "Rainy".

## 🏃 Getting Started
1. Clone this repository.
2. Install dependencies: ```pip install -r requirements.txt ```
3. Run the app: ```streamlit run main.py.```
