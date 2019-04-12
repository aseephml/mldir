class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed  #TypeError: 'LGBMClassifier' object does not support item assignment
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    #def predict_proba(self, x):
    #    return self.clf.predict_proba(x)    
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_) 

#import numpy as np, pandas as pd
#from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
#                              GradientBoostingClassifier, ExtraTreesClassifier)
#train_data = [[1, 4, 5, 6],
#              [4, 5, 6, 7],
#              [30, 40, 50, 60]]
#
#
#train_labels = [10, 20, 30]
#
#
#eval_data = [[2, 4, 6, 8],
#             [1, 4, 50, 60]]
#
#dftrX = pd.DataFrame(train_data)
#dftry = pd.DataFrame(train_labels)
#dfteX = pd.DataFrame(eval_data)
#        
#et = SklearnHelper(clf=ExtraTreesClassifier, params={})
#mdl = et.fit(dftrX, train_labels) 
#pred = mdl.predict(dfteX)       