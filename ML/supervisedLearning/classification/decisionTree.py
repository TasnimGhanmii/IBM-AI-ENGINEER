import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

#transforming categorical data into numerical ones

#note: we used label encoder because it is typically used for encoding target variables (labels) or when the categorical feature has an ordinal relationship (i.e., there is a meaningful order to the categories)
#for example: [cat,dog]=>[0,1]
# but one hot encoder  it is typically used for encoding categorical features where there is no ordinal relationship between the categories. This prevents the model from assuming any order or hierarchy in the categories.
#for example: [cat,dog]=>[[1,0],[0,1]]

label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

#check if there's any missing data
#my_data.isnull().sum()

#mapping drugs to numerical values to be able to evaluate the correation of the target variable with the features
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
my_data.drop('Drug',axis=1).corr()['Drug_num']

#preparing data
y=my_data['Drug']
X=my_data.drop(['Drug','Drug_num'],axis=1)

X_trainset,X_testset,y_trainset,y_testset=train_test_split(X,y,test_size=0.3,random_state=32)

#model
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)
tree_predictions = drugTree.predict(X_testset)
print('accuracy',metrics.accuracy_score(y_testset,tree_predictions))

#visualize
plot_tree(drugTree)
plt.show()
