""" a binary classfication and prediction example
by using logistic regression (sklearn) """

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
from sklearn.metrics import roc_auc_score


# read input data from program or from csv file
def load_data(path):
    exists = os.path.isfile(path)
    if not exists:
        d = {'age':[22,25,47,52,46,56,55,60,62,61,18,28,27,29,49,55,25,58,19,18,21,26,40,45,50,54,23],'bought_insurance':[0,0,1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0]}
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False, encoding='utf-8')
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path)
    return df


def train_and_predict():
    df = load_data(sys.argv[1])

    # split into train and test dataset
    x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance, test_size=0.1)

    # train and predict
    model = LogisticRegression()
    model.predict(x_test)
    #model.predict([[70]])

    # model quality
    print("Training set score: {:.3f}".format(model.score(x_train,y_train)))
    print("Testing set score: {:.3f}".format(model.score(x_test,y_test)))

    def lr_model(x):
        return 1 / (1 + np.exp(-x))
    # plot the loss of the test data
    loss = lr_model(x_test * model.coef_ + model.intercept_).values.ravel()
    plt.plot(x_test, loss, color='black', linewidth=3)
    sorted_input = df[['age']].sort_values(by=['age'])
    whole_prediction = model.predict(sorted_input)
    # plot the logistic model result
    plt.plot(sorted_input, whole_prediction, color='red', linewidth=3 )
    # scatter plot of the input data
    plt.scatter(df.age, df.bought_insurance, marker='+', color='blue')
    plt.show()

    #Plot the cumulative gains graph (I used training+testing data due to the small data size, ideally only test data is used)
    roc_auc_score(sorted_input, whole_prediction)
    skplt.metrics.plot_cumulative_gain(sorted_input, whole_prediction)
    plt.show()
    skplt.metrics.plot_lift_curve(true_values, predictions)
    plt.show()

"""
    # the logit function from statsmodels provide summary to help with model optimization
    import statsmodels.api as sm
    logit_model=sm.Logit(y_train,x_train)
    result=logit_model.fit()
    print(result.summary())
"""

if __name__=="__main__":
    train_and_predict()
