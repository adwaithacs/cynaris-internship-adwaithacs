import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("day2_dataset.csv")
df.head()
df.info()
df.describe()
#histogram
plt.figure(figsize=(8, 5))
plt.hist(df["Total Cases"], bins=10)
plt.title("Distribution of Total Cases")
plt.xlabel("Total Cases")
plt.ylabel("Number of States/UTs")
plt.show()
#bar chart
plt.plot(df["State/UTs"], df["Total Cases"])
plt.title("Total Cases by State/UT")
plt.xlabel("State/UT")
plt.ylabel("Total Cases")
plt.show()
#scatter plot
plt.scatter(df["Population"], df["Total Cases"])
plt.show()
#box plot
df[["Active", "Discharged", "Deaths"]].plot(kind="box")
plt.title("COVID-19 Cases Distribution")
plt.ylabel("Number of Cases")
plt.show()
#line chart
plt.plot(df["State/UTs"], df["Total Cases"])
plt.xticks(rotation=45)
plt.title("Total Cases by State/UT")
plt.xlabel("State/UT")
plt.ylabel("Total Cases")
plt.show()
#pie chart
df[["Active", "Discharged", "Deaths"]].sum().plot(kind="pie", autopct="%1.1f%%")
plt.title("COVID-19 Case Distribution")
plt.ylabel("")
plt.show()
#seaborn heatmap
import matplotlib.pyplot as plt
corr = df[["Total Cases", "Active", "Discharged", "Deaths", "Population"]].corr()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()