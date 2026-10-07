import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)
df = pd.read_csv("day4_dataset.csv")
print(df)
df.shape
df.head(5)
df.columns
df.dtypes
df.describe()
print(df.columns.tolist())
df.info()
df.isnull().sum()
#numeric column
numeric_columns = df.select_dtypes(include=np.number).columns
numeric_columns
df[numeric_columns].hist(bins=20,figsize=(12, 10))
plt.tight_layout()
plt.show()
#correlation heatmap
df_corr = df.select_dtypes(include=np.number).corr()
df_corr
sns.heatmap(df_corr, annot=True)
plt.title('Titanic Correlation Heatmap')
plt.show()
#10 category counts
df['Embarked'].value_counts().head(10)
ax = df['Embarked'].value_counts().head(10).plot(kind='bar',title='Top 10 Embarked Categories')
ax.set_xlabel('Embarked')
ax.set_ylabel('Count')
plt.show()
df.info()
df.isnull().sum()
# EDA Narrative
#The Titanic dataset contains 891 passenger records and 12 columns.
#The dataset includes numerical and categorical features such as passenger
#class, age, sex, family size, fare, cabin, and embarkation point.
#The distribution plots show the spread of the numerical variables. Age and Fare have continuous values, while variables such as SibSp and Parch contain integer counts. The categorical analysis shows the frequency of values in features such as Embarked.
#The correlation heatmap helps identify relationships between the numerical features. These relationships show how variables are associated with each other, but they do not indicate causation.
#The dataset also contains missing values that need attention before machine learning. Age has 714 non-null values, meaning some age information is missing. Cabin has only 204 non-null values, making it the column with a large amount of missing information. Embarked has 889 non-null values, so small number of entries are missing there. These missing values should be handled appropriately during preprocessing.Overall, the Titanic dataset provides useful passenger information for EDA,
#but missing values and data quality should be addressed before using the dataset to build a machine learning model.