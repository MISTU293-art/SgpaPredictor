from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

Hours = np.array([
    [1],[2],[2],[3],[3],[4],[4],[5],[5],[6],
    [6],[7],[7],[8],[8],[9],[9],[10],[10],[11],
    [11],[12],[12],[13],[13],[14],[14],[15],[15],[16],
    [16],[17],[17],[18],[18],[19],[19],[20],[20],[21],
    [21],[22],[22],[23],[23],[24],[24],[25],[25],[26]
])

SGPA = np.array([
    [5.2],[5.4],[5.6],[5.8],[6.0],[6.1],[6.2],[6.4],[6.5],[6.7],
    [6.8],[7.0],[7.1],[7.2],[7.3],[7.4],[7.5],[7.6],[7.7],[7.8],
    [7.9],[8.0],[8.1],[8.2],[8.3],[8.4],[8.5],[8.6],[8.7],[8.8],
    [8.85],[8.9],[8.95],[9.0],[9.02],[9.05],[9.08],[9.1],[9.12],[9.15],
    [9.18],[9.2],[9.22],[9.25],[9.28],[9.3],[9.32],[9.35],[9.38],[9.4]
])

model=LinearRegression()
X_train,x_test,Y_train,y_test=train_test_split(
    Hours,SGPA,test_size=0.2,random_state=42
)
model.fit(X_train,Y_train)
hours=float(input("Enter Your Expected SGPA:"))
score=model.score(Hours,SGPA)
prediction=model.predict([[hours]])

print(f"Model Score:{score:.2f}")
print(f"Required Hours To Score:{prediction[0][0]:.2}")