
# Import necessary libraries for data manipulation and model development
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('car_data.csv')

# Display the first few rows to understand the dataset structure
print("First rows of the dataset:")
print(df.head())

# Check the data types and null values
print("
Dataset information:")
print(df.info())

# Convert 'DateCrawled' to datetime and extract useful features like hour and day of the week
df['DateCrawled'] = pd.to_datetime(df['DateCrawled'])
df['CrawledHour'] = df['DateCrawled'].dt.hour
df['CrawledDayOfWeek'] = df['DateCrawled'].dt.dayofweek
df.drop('DateCrawled', axis=1, inplace=True)

# Handle missing values in categorical columns with 'unknown' and in numerical columns with the median
categorical_columns = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'NotRepaired']
for column in categorical_columns:
    df[column].fillna('unknown', inplace=True)

numerical_columns = ['Power']
for column in numerical_columns:
    df[column].fillna(df[column].median(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encoding categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Split the data into features (X) and target variable (y)
X = df_encoded.drop(['Price', 'DateCreated', 'LastSeen', 'PostalCode', 'NumberOfPictures'], axis=1)
y = df_encoded['Price']

# Split the data into training, validation, and test sets (60% train, 20% validation, 20% test)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

# Train a Linear Regression model
start_time = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_train_time = time.time() - start_time

# Train a RandomForest Regressor
start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

# Train a LightGBM model
start_time = time.time()
lgbm_model = LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1)
lgbm_model.fit(X_train, y_train)
lgbm_train_time = time.time() - start_time

# Evaluate models on validation set
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

lr_rmse = evaluate_model(lr_model, X_val, y_val)
rf_rmse = evaluate_model(rf_model, X_val, y_val)
lgbm_rmse = evaluate_model(lgbm_model, X_val, y_val)

print(f"Linear Regression RMSE: {lr_rmse}, Training Time: {lr_train_time:.2f} seconds")
print(f"Random Forest RMSE: {rf_rmse}, Training Time: {rf_train_time:.2f} seconds")
print(f"LightGBM RMSE: {lgbm_rmse}, Training Time: {lgbm_train_time:.2f} seconds")

# Find the best model and test it on the test set
best_model = lgbm_model if lgbm_rmse < lr_rmse and lgbm_rmse < rf_rmse else lr_model if lr_rmse < rf_rmse else rf_model

best_model_rmse = evaluate_model(best_model, X_test, y_test)
print(f"Best Model RMSE on Test Set: {best_model_rmse}")