import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df=pd.read_csv(url)
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

churn_df['churn'] = churn_df['churn'].astype('int')

#here we convert to numpy array cause most scikitlearn algos are built on top of numpy
#although sometimes sklearn accepts dataframaes it coverts them to numpy arrays
#explicitly doing this make it clear and helps avoid potential
#features
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
#target
y = np.asarray(churn_df['churn'])
#standarizng features
X_norm = StandardScaler().fit(X).transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_norm,test_size=0.2,random_state=4)

LR=LogisticRegression().fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

