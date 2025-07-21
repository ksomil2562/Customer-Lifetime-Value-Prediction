
import pandas as pd

# Load the Excel file from Desktop
df = pd.read_excel(r'C:\Users\kumar\Desktop\Online Retail.xlsx')

# Show first 5 rows
print("First 5 rows of the data:")
print(df.head())

# Show column names and data types
print("\nColumn Info:")
print(df.info())

# Summary statistics
print("\nData Summary:")
print(df.describe())



from clean_data import clean_online_retail_data

df_clean = clean_online_retail_data(df)
print("\n Cleaned Data Preview:")
print(df_clean.head())
print(f"\nTotal rows after cleaning: {len(df_clean)}")


from rfm_analysis import calculate_rfm

rfm_df = calculate_rfm(df_clean)
print("\n RFM Table:")
print(rfm_df.head())

print(f"\nTotal customers in RFM table: {len(rfm_df)}")


from clv_model import train_clv_model

model, y_test, y_pred = train_clv_model(rfm_df)
