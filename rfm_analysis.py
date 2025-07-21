import pandas as pd

# Load cleaned data
df_clean = pd.read_csv("cleaned_data.csv")  # Make sure this file exists

# Step 1: Calculate RFM
def calculate_rfm(df_clean):
    # Use latest date in the dataset as "today"
    reference_date = pd.to_datetime(df_clean['InvoiceDate']).max()

    # Group by CustomerID to calculate Recency, Frequency, Monetary
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - pd.to_datetime(x).max()).days,  # Recency
        'InvoiceNo': 'nunique',                                                    # Frequency
        'TotalPrice': 'sum'                                                        # Monetary
    }).reset_index()

    # Rename columns
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Save to CSV
    rfm.to_csv("rfm_data.csv", index=False)

    return rfm

# Run RFM calculation
rfm = calculate_rfm(df_clean)

# Preview result
print("Top 5 RFM rows:")
print(rfm.head())
