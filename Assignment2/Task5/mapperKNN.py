#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from collections import Counter

def KNN(X_train, X_test, Y_train):
    num_test=X_test.shape[0]
    num_train = X_train.shape[0]
    #print(X_test.shape, X_train.shape)
    distance = np.zeros((num_test,num_train))
    #print("dis init ", distance)
    distance = np.sqrt(np.sum(X_test**2, axis=1).reshape(num_test, 1) + np.sum(X_train**2, axis=1) - 2 * X_test.dot(X_train.T))

    return distance 

count = 0
for line in sys.stdin:
        #print("\n\n\n start")
        count +=1
        x_test = pd.read_csv(r'./Test.csv')
        #x_actual = data.iloc[:, :-1]
        x_test = np.asarray(x_test, dtype=np.float32)
        #y_actual = data.iloc[:, -1]
        #print(x_actual.shape)
        #print(line)
        words=line.strip().split(",")
        #print(words, len(words))
        y_actual = words[-1]
        words = words[:48]
        num = np.asarray(words, dtype=np.float32)
        num = np.reshape(num, (48,1))
        #print(num.shape, type(num))
        cat = KNN(np.transpose(num), x_test, y_actual)
        #print("Answer ", cat, "shape ", cat.shape)
        primaryKey = 1
        #print(primaryKey)
        for x in np.nditer(cat):
            print('%s\t%s\t%s'%((str(primaryKey)), x, y_actual))
            primaryKey += 1

