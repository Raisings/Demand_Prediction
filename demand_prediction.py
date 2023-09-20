import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

### loading speed will be faster when using local dataset over online ###
data = pd.read_csv("demand.csv")
# data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
print(data.head())

## check if the dataset contains any null values
print(data.isnull().sum())
print("no. of cells with NULL value : ", data.isnull().sum().sum())
print("no. of rows : ", len(data))

## remove that row with missing value in the Total Price column using dropna()
data = data.dropna()
print(data.isnull().sum())
print("no. of rows after cleaning : ", len(data))

## analyzing relationship between the price and demand using scatter plot
fig = px.scatter(data, x="Units Sold", y="Total Price", size='Units Sold')
# fig.show()

## see the correlation between the features of the dataset
print(data.corr())
correlations = data.corr(method="pearson")
plt.figure(figsize=(15, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
# plt.show()

## training a machine learning model
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(xtrain.values, ytrain.values) ## use .value here as it would send df as input to fit with headers without using .values

features = np.array([[133.00, 140.00]])
print(model.predict(features))
