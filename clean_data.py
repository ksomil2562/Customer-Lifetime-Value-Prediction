import pandas as pd

# Load Excel file
df = pd.read_excel("Online Retail.xlsx")

# Drop rows with missing CustomerID or InvoiceNo
df_clean = df.dropna(subset=['CustomerID', 'InvoiceNo']).copy()

# Convert types safely
df_clean.loc[:, 'CustomerID'] = df_clean['CustomerID'].astype(int)
df_clean.loc[:, 'InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# Calculate total price
df_clean.loc[:, 'TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Keep only valid rows (positive quantity and price)
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Save to CSV
df_clean.to_csv("cleaned_data.csv", index=False)

# Print confirmation (use regular print to avoid encoding issues)
print("Cleaned data saved as cleaned_data.csv")
