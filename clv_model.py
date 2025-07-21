from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

def train_clv_model(rfm_df):
    # Features and target
    X = rfm_df[['Recency', 'Frequency', 'Monetary']]
    y = rfm_df['Monetary']  # We're predicting total spend as a proxy for CLV

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n MAE (Mean Absolute Error): {mae:.2f}")

    return model, y_test, y_pred
