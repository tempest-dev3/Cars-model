from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rich import print

df =pd.read_csv('car_prices.csv')
Year = np.array(df['year'])
Mileage = np.array(df['mileage'])
Engine_size = np.array(df['engine_size'])
Horsepower = np.array(df['horsepower'])
Doors = np.array(df['doors'])
Owners = np.array(df['owners'])
X = np.column_stack((Year, Mileage, Engine_size, Horsepower, Doors, Owners))
y = np.array(df['price'])

x_train, x_test ,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred= model.predict(x_test)
# print(f"b0 : {model.intercept_}")
# print(f"b1 : {model.coef_}")
y_test_pred = model.predict(x_test)
print("Predicted Prices:", y_pred[0])
print("[bold red]R2:", r2_score(y_test, y_test_pred))

plt.scatter(y_test, y_test_pred, color='blue', label='Predicted')
plt.xlabel('Actual Price',color='green')
plt.ylabel('Predicted Price',color='red')


plt.show()
# print("Actual Prices:", y_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# print("[bold red]Mean Absolute Error:", mae)
# print("[bold red]Mean Squared Error:", mse)
# print("[bold red]Root Mean Squared Error:", rmse)