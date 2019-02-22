""" a binary classfication and prediction example
by using logistic regression (sklearn) """

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklean.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

# read input data from program or from csv file
d = {'age':[22,25,47,52,46,56,55,60,62,61,18,28,27,29,49,55,25,58,19,18,21,26,40,45,50,54,23],'bought_insurance':[0,0,1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0]}
df = pd.DataFrame(data=d)
df.to_csv('insurance_data.csv', index=False, encoding='utf-8')
df = pd.read_csv('insurance_data.csv')

# scatter plot of the input data
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.show()

# split into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance, test_size=0.1)

# train and predict
model = LogisticRegression()
model.fit(x_train,y_train)
model.predict(x_test)
model.score(x_test,y_test)
model.predict_proba(x_test)
model.predict([[70]])
