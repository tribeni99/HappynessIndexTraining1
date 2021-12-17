import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing
# %%
df = pd.read_csv('G:/Data/mushrooms.csv')

# %%
temp = 0
for i in df.columns:
    le = preprocessing.LabelEncoder()
    le.fit(df[i])
    li = list(le.classes_)
    regList = le.transform(df[i])
    
    df[i] = regList
    if temp == 0:
        tempDf = df
        one_hot_encoded_data = pd.get_dummies(tempDf, columns = [i])
    else:
        tempDf = one_hot_encoded_data
        one_hot_encoded_data = pd.get_dummies(tempDf, columns = [i])
    temp += 1
    print(i)
    # input()
    # one_hot_encoded_data.drop('Reg', axis = 1)
# %%
x = np.array(one_hot_encoded_data)[:, 2:]
y = np.array(one_hot_encoded_data)[:,0]
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)

# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
# %%
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)

print ("Confusion Matrix : \n", cm)


# %%