import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
columns_to_keep = ['LotArea', 'BedroomAbvGr','BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF',  'FullBath', 'SalePrice']
train = train[columns_to_keep]
test = test[['LotArea', 'BedroomAbvGr','BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF',  'FullBath']] 
train['SalePrice'] = np.log1p(train['SalePrice'])
new_skewness = skew(train['SalePrice'])
print("Skewness after logarithmic transformation:", new_skewness)
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
imputer = SimpleImputer(strategy='mean')
X_test_scaled = imputer.fit_transform(X_test_scaled)
ridge = Ridge(alpha=1.0) 
ridge.fit(X_train_scaled, y_train)
predictions = ridge.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
plt.scatter(np.expm1(y_test), np.expm1(predictions))
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs. Predicted Sale Price')
plt.show()
test_scaled = scaler.transform(test)
test_scaled = imputer.transform(test_scaled)
test_predictions = ridge.predict(test_scaled)
predicted_sale_price = np.expm1(test_predictions)
plt.scatter(test['LotArea'], predicted_sale_price)
plt.xlabel('Lot Area')
plt.ylabel('Predicted Sale Price')
plt.title('Lot Area vs. Predicted Sale Price')
plt.show()
