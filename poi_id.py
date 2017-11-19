#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import tester
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
%matplotlib inline

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',
                 'bonus','deferral_payments',
                 'deferred_income','director_fees',
                 'exercised_stock_options','expenses',
                 'loan_advances', 'long_term_incentive',
                 'other','restricted_stock',
                 'restricted_stock_deferred', 'total_payments',
                 'total_stock_value','from_messages',
                 'from_poi_to_this_person','from_this_person_to_poi',
                 'shared_receipt_with_poi','to_messages'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
###Dataset Exploration
data_dict_df = pd.DataFrame.from_dict(data_dict, orient = 'index')
data_dict_df = data_dict_df.replace('NaN', np.nan)
data_dict_df.info()
data_dict_df['poi'].value_counts()

### Task 2: Remove outliers
def plotoutliers(data_dict, feature_x, feature_y):
    data = featureFormat(data_dict, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]        
        plt.scatter(x, y)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

plotoutliers(data_dict, 'salary', 'bonus')
plotoutliers(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
plotoutliers(data_dict, 'total_payments', 'total_stock_value')
plotoutliers(data_dict, 'deferred_income','director_fees')
plotoutliers(data_dict, 'from_messages', 'to_messages')
plotoutliers(data_dict, 'restricted_stock', 'restricted_stock_deferred')



print (data_dict['LAY KENNETH L']['poi'])
print(data_dict['WHITE JR THOMAS E']['poi'])
print(data_dict['LOCKHART EUGENE E']['poi'])

###Therefore, the one with extremely high salary, bonus, director_fee and extremely low deferred_income is the outlier, which is "total". Lay,KENNETH L has super high total_payments, total_stock_value and restricted_stock, but we see it's actually person of interest, so we'll not remove it. WHITE JR, THOMAS E has super high restricted_stock, so it's also an outlier Besides, THE TRAVEL AGENCY IN THE PARK is not a person's name, so it's also should be removed. Also, There's no data for LOCKHART, EUGENE E, so it should also be removed.


outliers=['TOTAL','WHITE JR THOMAS E','LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']
for key in outliers:
    data_dict.pop(key)
    

### Features' Scores for the Original Feature Set 
features_list
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = MinMaxScaler()
skb = SelectKBest(k = 'all')
clf = GaussianNB()
pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('GaussianNB',clf)])
pipeline.fit(features, labels)

skb = pipeline.named_steps['skb']


feature_scores = ['%.2f' % elem for elem in skb.scores_ ]
feature_scores_pvalues = ['%.2f' % elem for elem in  skb.pvalues_ ]
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in skb.get_support(indices=True)]

features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple
    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#I'm going to add the two new features, which are 'messages_from_poi_ratio' and 'messages_to_poi_ratio'.
def messages_with_poi_ratios(data_dict):
    for key in data_dict:
        messages_from_poi = data_dict[key]['from_poi_to_this_person']
        to_messages = data_dict[key]['to_messages']
        if messages_from_poi != "NaN" and to_messages != "NaN":
            data_dict[key]['messages_from_poi_ratio'] = float(messages_from_poi)/float(to_messages)
        else:
            data_dict[key]['messages_from_poi_ratio'] = 0
        messages_to_poi = data_dict[key]['from_this_person_to_poi']
        from_messages = data_dict[key]['from_messages']
        if messages_to_poi != "NaN" and from_messages != "NaN":
            data_dict[key]['messages_to_poi_ratio'] = float(messages_to_poi)/float(from_messages)
        else:
            data_dict[key]['messages_to_poi_ratio'] = 0

messages_with_poi_ratios(data_dict)  
my_dataset = data_dict
features_list = features_list + ['messages_from_poi_ratio', 'messages_to_poi_ratio']
features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print(features_list)
print(labels)


### Features' Scores for the New Feature Set 
scaler = MinMaxScaler()
skb = SelectKBest(k = 'all')
clf = GaussianNB()
pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('GaussianNB',clf)])
pipeline.fit(features, labels)

skb = pipeline.named_steps['skb']


feature_scores = ['%.2f' % elem for elem in skb.scores_ ]
feature_scores_pvalues = ['%.2f' % elem for elem in  skb.pvalues_ ]
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in skb.get_support(indices=True)]

features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#Feature Scaling and Feature Selection
scaler = MinMaxScaler()
skb = SelectKBest(k = 7)

print ('Gaussian Naive Bayes')
clf = GaussianNB()
pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('GaussianNB',clf)])
tester.test_classifier(pipeline, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')

print ('XGBoost')
clf = XGBClassifier()
pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('XGBoost',clf)])
tester.test_classifier(pipeline, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')

print ('Decision Tree')
clf = DecisionTreeClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('dtree',clf)])
tester.test_classifier(pipeline, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')

print('Random Forest')
clf = RandomForestClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('randomf',clf)])
tester.test_classifier(pipeline, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')

print('Logistic Regression')
clf = LogisticRegression()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('logire',clf)])
tester.test_classifier(pipeline, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')


print('K-nearest neighbors')
clf = KNeighborsClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('knn',clf)])
tester.test_classifier(pipeline, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')


print('Ada Boost')
clf = AdaBoostClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('adab',clf)])
tester.test_classifier(pipeline, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print ('Tune Logistic Regression')
clf = LogisticRegression()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('logire',clf)])

parameters = {'skb__k':range(16,19),
                 'logire__C': [0.001,0.01, 0.1, 1],
                 'logire__tol': [1e-2, 1e-3, 1e-4,1e-5],
                 'logire__penalty': ['l1', 'l2']
                 
              }




sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 0)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
tester.test_classifier(clf, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')


print ('Tune XGBoost')
clf = XGBClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('xgb',clf)])

parameters = {'skb__k':range(7,10),              
              'xgb__max_depth':[3,6]
                 
              }




sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 0)

grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')

grid_search.fit(features, labels)
clf = grid_search.best_estimator_

tester.test_classifier(clf, my_dataset, features_list)



print ('Tune Decision Tree')
clf = DecisionTreeClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('dtree',clf)])


parameters = {'skb__k':range(7,15),
              'dtree__criterion':('gini','entropy'),
              'dtree__splitter':('best','random'),
                 
              }

sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 0)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
tester.test_classifier(clf, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')



print ('Tune Random Forest')
clf = RandomForestClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('randomf',clf)])


parameters = {'skb__k':range(7,15),    
              'randomf__criterion':('gini','entropy')               
              }

sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 0)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
tester.test_classifier(clf, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')


print ('Tune K-nearest neighbors')
clf = KNeighborsClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('knn',clf)])


parameters = {'skb__k':range(7,15),
              'knn__n_neighbors':[2,5,7],
              'knn__algorithm':('auto','ball_tree','kd_tree','brute')
                
              }

sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 0)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
tester.test_classifier(clf, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')




print ('Tune AdaBoost')
clf = AdaBoostClassifier()

pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('adab',clf)])

parameters = {'skb__k':range(10,15),
              'adab__n_estimators':(50,200,500)
          
                 
              }

sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 0)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
tester.test_classifier(clf, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')





print ('Tune Gaussian Naive Bayes')
clf = GaussianNB()
pipeline=Pipeline([('min_max_scaler',scaler),
       ('skb',skb),
       ('GaussianNB',clf)])



parameters = {'skb__k':range(7,12)
                 
              }

sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 0)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
tester.test_classifier(clf, my_dataset, features_list)
print ('---------------------------------------------------------------------------------------------------------------------')

data_dict_df['poi'].value_counts().plot(kind='bar')

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)