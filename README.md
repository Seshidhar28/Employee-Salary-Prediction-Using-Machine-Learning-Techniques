# Results - Employee Salary Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6],
    'Education_Level': [2, 2, 3, 3, 4, 4],
    'Salary': [30000, 35000, 40000, 45000, 50000, 55000]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Experience', 'Education_Level']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Actual vs Predicted
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\n📋 Actual vs Predicted:")
print(results_df)
