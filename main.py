%pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = np.array([
    [30,70],
    [25,80],
    [27,60],
    [31,65],
    [23,85],
    [28,75]
])
y = np.array([0,1,0,0,1,1])
new_weather = np.array([[26,78]])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)

pred = knn.predict(new_weather)[0]

label_map = {
    0:"Sunny",
    1:"Rainy"
}
plt.scatter(x[y==0,0], x[y==0,1], color="orange", label="Sunny", s=100, edgecolor="k")
plt.scatter(x[y==1,0], x[y==1,1], color="blue", label="Rainy", s=100, edgecolor="k")
colors = ["orange", "blue"]
plt.scatter(new_weather[0,0], new_weather[0,1],
            color= colors[pred], marker= "*", s=300, edgecolor= "black", label= f"New day: {label_map[pred]}")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Weather Classification")
plt.legend()
plt.grid(True)
plt.show()
