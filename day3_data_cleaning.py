import pandas as pd
import numpy as np
#load dataset
df= pd.read_csv("day3_student_data.csv")
print(df)
#handling missing data using dropna()
df1 = df.dropna(axis=0, how="any")
print(df1)
#drop coloumns where all values are NaN
df2 = df.dropna(axis=1, how="all")
print(df2)
#specific column
df3 = df.dropna(subset=["Score"])
print(df3)
#fill all missing value with asingle value
df5 = df.dropna(subset=['Student_ID', 'Grade'])
print(df5)
#handle missing data using fillna()
#Fill all missing values with one value
df6 = df.fillna(0)
print(df6)
#Forward fill
df7 = df.ffill()
print(df7)
#Backward fill with limit
df8 = df.bfill(limit=1)
print(df8)
#Fill with a different value
df9 = df.fillna({'Score': 0,'Name': 'Unknown','Grade': 'Not Assigned'})
print(df9)
#removing duplicates using drop_duplicates
#Remove all duplicate rows
df10 = df.drop_duplicates()
print(df10)
#Remove duplicates based on specific columns
df11 = df.drop_duplicates(subset=['Student_ID'])
print(df11)
#keep last duplicate
df12 = df.drop_duplicates(subset=['Student_ID'], keep='last')
print(df12)
#replace values
df13 = df.replace({'A': 'Excellent','B': 'Good','C': 'Average'})
print(df13)
#type conversion
df14['Score'] = df14['Score'].astype(float)
#Trim whitespace
df15 = df["Name"].str.strip()
