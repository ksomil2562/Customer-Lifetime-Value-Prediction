import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# Step 1: Load data and model
rfm = pd.read_csv("rfm_data.csv")
model = joblib.load("clv_model.pkl")

# Step 2: Predict CLV
rfm['predicted_clv'] = model.predict(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 3: If you have actual CLV, compare it (optional)
# Example: Uncomment if you have actual values
# mae = mean_absolute_error(rfm['ActualCLV'], rfm['predicted_clv'])
# print(f" MAE: {round(mae, 2)}")

# Step 4: Identify top 20% high-value customers
rfm_sorted = rfm.sort_values('predicted_clv', ascending=False)
top_20_percent = rfm_sorted.head(int(0.2 * len(rfm)))
top_20_revenue = top_20_percent['predicted_clv'].sum()
total_revenue = rfm['predicted_clv'].sum()
contribution_pct = round((top_20_revenue / total_revenue) * 100, 2)

# Step 5: Estimate Revenue Opportunity
# Let's say you improve retention of these top customers by 20%
expected_gain = 0.20 * top_20_revenue

# Step 6: Save predictions for dashboard or reports
rfm.to_csv("clv_predictions.csv", index=False)

# Final Summary
print(" CLV Prediction Completed")
print(f" Top 20% customers contribute ~{contribution_pct}% of predicted revenue.")
print(f" Estimated ₹{round(expected_gain, 2)} in additional lifetime revenue with targeted retention.\n")
print(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'predicted_clv']].head())
