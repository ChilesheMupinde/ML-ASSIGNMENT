from pickle import dump

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("supermarket_sales.csv")

features = df[[ 'gross_income', 'total_cost', 'cogs']]
target = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linearmodel = LinearRegression()

linearmodel.fit(X_train,y_train)

dump(linearmodel, open('model.pkl', 'wb'))

