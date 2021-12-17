import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing
# %%
df = pd.read_csv('G:\Data\happiness_score_dataset.csv')

# %%
le = preprocessing.LabelEncoder()
le.fit(df['Region'])
li = list(le.classes_)
regList = le.transform(df['Region'])

df['Reg'] = regList

one_hot_encoded_data = pd.get_dummies(df, columns = ['Reg'])
# one_hot_encoded_data.drop('Reg', axis = 1)
# %%
X = np.array(df[list(df.columns)[4:]])
Y = np.array(df['Happiness Score'])

ch = np.random.choice(range(158), 10, replace=False)

X_Train = []
Y_Train = []
X_Test = []
Y_Test = []
for i in range(len(X)):
    if i not in ch:
        X_Train.append(X[i])
        Y_Train.append(Y[i])
    else:
        X_Test.append(X[i])
        Y_Test.append(Y[i])         
        
# X = np.array(df[list(df.columns)[4:]][:-10])
# Y = np.array(df['Happiness Score'][:-10])

# X_Test = np.array(df[list(df.columns)[4:]][-10:])
# Y_Test = np.array(df['Happiness Score'][-10:])
# %%

regr = linear_model.LinearRegression()
regr.fit(X_Train, Y_Train)

# %%
for i in range(len(X_Test)):
    pred = regr.predict(np.array([X_Test[i]]))
    
    print(pred, np.array(Y_Test[i]))


# %%