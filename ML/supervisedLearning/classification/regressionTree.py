import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

#correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
#correlation_values.plot(kind='barh', figsize=(10, 6))

#data preprocessing
                          #transforms a dataframe into a numpy array
y=raw_data[['tip_amount']].values.astype('float32')
proc_data=raw_data.drop(['tip_amount'],axis=1)
X=proc_data.values
X=normalize(X,axis=1,norm='l1',copy=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regTree=DecisionTreeRegressor(criterion='squared_error',max_depth=8,random_state=35)
regTree.fit(X_train,y_train)

y_pred=regTree.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print('MSE : {0:.3f}'.format(mse))

r2_score =regTree.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))