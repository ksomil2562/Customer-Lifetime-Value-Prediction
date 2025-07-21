import pandas as pd
import joblib

# Step 1: Load the saved RFM data and trained model
rfm = pd.read_csv("rfm_data.csv")
model = joblib.load("clv_model.pkl")

# Step 2: Predict CLV using the model
rfm['predicted_clv'] = model.predict(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 3: Save predicted CLV to CSV
rfm.to_csv("predicted_clv.csv", index=False)

# Step 4: Display top 10 customers by CLV
print("Top customers by predicted CLV:")
print(rfm[['CustomerID', 'predicted_clv']].sort_values(by='predicted_clv', ascending=False).head(10))


import matplotlib.pyplot as plt
import seaborn as sns

# Step 5: Plot CLV distribution
plt.figure(figsize=(10, 6))
sns.histplot(rfm['predicted_clv'], bins=50, kde=True, color="skyblue")
plt.title("Distribution of Predicted Customer Lifetime Value (CLV)")
plt.xlabel("Predicted CLV")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.tight_layout()
plt.savefig("clv_distribution.png")  # Save image
plt.show()

# Step 6: Plot Top 10 customers by CLV
top_customers = rfm.sort_values(by='predicted_clv', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_customers, x='CustomerID', y='predicted_clv', palette="viridis")
plt.title("Top 10 Customers by Predicted CLV")
plt.xlabel("Customer ID")
plt.ylabel("Predicted CLV")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("top_customers.png")  # Save image
plt.show()
