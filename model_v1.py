import pandas as pd
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
sampregdata = pd.read_csv("sampregdata.csv")
best_X = sampregdata[['x4']]
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(best_X, y, test_size = 0.2) 
model_best = LinearRegression()
model_best.fit(X_train_best, y_train_best)
y_pred_best = model_best.predict(X_test_best)
mse_best = mean_squared_error(y_test_best, y_pred_best)
r2_best = r2_score(y_test_best, y_pred_best)
print(f"The MSE for the best model is: {mse_best}, The R² is: {r2_best:.4f}")