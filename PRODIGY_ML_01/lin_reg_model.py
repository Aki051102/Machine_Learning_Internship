'''TASK 1: Implement a linear regression model to predict the prices of houses based on their
square footage and the number of bedrooms and bathrooms.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset_path = r'C:\Users\Akash\Desktop\Machine_Learning_Internship\Machine_Learning_Internship\PRODIGY_ML_01\dataset\train.csv'
df = pd.read_csv(dataset_path)

print(df.head())
print(df.info())
print(df.describe())

df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

print(df.isnull().sum())

df = df.dropna()

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
