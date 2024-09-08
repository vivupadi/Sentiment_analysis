from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV 

class models:
    def __init__(self):
        pass

    def log_reg(self):
        model =  LogisticRegression
        param = { 
        'solver': ['lbfgs', 'liblinear'],  
        'C': [10, 1.0, 0.1, 0.01] 
        }
        return model, param
    
    def XGB(self):
        model = XGBClassifier
        param = { 
        'min_child_weight': [1, 3 ],
        'gamma': [0.5, 1, 5],
        'subsample': [0.6],
        'colsample_bytree': [0.6, 1],
        'max_depth': [3, 5, 10]
        } 
        return model, param
    
    def RanFo(self):
        model = RandomForestClassifier
        param = { 
        'n_estimators': [25, 100, 500], 
        'max_features': ['sqrt'], 
        'max_depth': [3, 6, 9], 
        'max_leaf_nodes': [3, 6, 9], 
        } 
        return model, param