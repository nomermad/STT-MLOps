import pandas as pd
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
sampregdata = pd.read_csv("sampregdata.csv")
X2 = sampregdata[['x2','x4']]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size = 0.2) 
model2 = LinearRegression()
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)
mse2 = mean_squared_error(y_test2, y_pred2)
r2_2 = r2_score(y_test2, y_pred2)
print(r2_2)