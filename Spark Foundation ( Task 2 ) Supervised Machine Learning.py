import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df.head()
df.isna().sum()
df.shape
df.describe()
df.info()
%matplotlib inline
plt.xlabel('hours')
plt.ylabel('scores')
plt.scatter(df.Hours,df.Scores,color='green',marker='*')
sns.lmplot(x='Hours', y='Scores', data=df)
df.corr()
sns.heatmap(df.corr())
X = df.Hours.values  
y = df.Scores.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=20)
lr = LinearRegression()
lr.fit(df[['Hours']],df.Scores)
dir(lr)
print(lr.intercept_)
print(lr.coef_)
pred = lr.predict(df[['Hours']])
pred
lr.predict([[9.25]])
from sklearn import metrics
from sklearn.metrics import *
print('Root mean squared error:', np.sqrt(metrics.mean_squared_error(df.Scores, pred)))
print("r2_score :", r2_score(df.Hours,df.Scores))

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

