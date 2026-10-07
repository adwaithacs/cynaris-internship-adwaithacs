import pandas as pd
df = pd.read_csv("day2_dataset.csv")
#print shape
print("shape:", df.shape)
#print datatype
print("\nData Types:")
print(df.dtypes)
#print the first 10 rows
print("\nFirst 10 rows:")
print(df.head(10))
#filter states more than 1,00,000 total cases
filtered_df = df[df["Total Cases"] > 100000]
print("\nFiltered data:",filtered_df)
print(filtered_df)
#groupby
grouped_df = df.groupby("State/UTs")[["Total Cases", "Active", "Discharged", "Deaths"]].mean()
print("\nGrouped Data:",grouped_df)
print(grouped_df)
#merge
# Merge operation
state_data = pd.DataFrame({"State/UTs": df["State/UTs"], "Population": df["Population"]})
merged_df = pd.merge(df, state_data, on="State/UTs")
print("\nMerged Data:")
print(merged_df.head(10))
#pivot table
pivot_df = pd.pivot_table(df,values="Total Cases",index="State/UTs",aggfunc="sum")
print("\nPivot Table:")
print(pivot_df)
#parquet
import pandas as pd
df = pd.read_csv("day1_dataset.csv")
import os
# Export cleaned DataFrame to CSV
df.to_csv("cleaned_covid_data.csv", index=False)
# Export cleaned DataFrame to Parquet
df.to_parquet("cleaned_covid_data.parquet", index=False)
# Compare file sizes
csv_size = os.path.getsize("cleaned_covid_data.csv")
parquet_size = os.path.getsize("cleaned_covid_data.parquet")
print("\nFile Size Comparison:")
print("CSV size:", csv_size, "bytes")
print("Parquet size:", parquet_size, "bytes")
#display no of rows after cleaning
print("\nNumber of rows:" , len(df))
# Display final dataset information
print("\nFinal Dataset Information:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])