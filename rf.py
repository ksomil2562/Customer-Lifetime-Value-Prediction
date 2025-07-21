import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Step 1: Load RFM data
df = pd.read_csv("rfm_data.csv")

# Step 2: Features and target
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['Monetary']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print(f" R² Score: {r2_score(y_test, y_pred):.2f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# Step 6: Save model
joblib.dump(model, "clv_model.pkl")
print(" Model saved as clv_model.pkl")
