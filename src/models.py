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
        'C': [10, 1.0, 0.1] 
        }
        return model, param
    
    def XGB(self):
        model = XGBClassifier
        param = { 
        'n_estimators': [25, 100, 150], 
        'max_features': ['sqrt', 'log2', None], 
        'max_depth': [3, 6, 9], 
        'max_leaf_nodes': [3, 6, 9], 
        } 
        return model, param
    
    def RanFo(self):
        model = RandomForestClassifier
        param = { 
        'n_estimators': [25, 50, 100, 150], 
        'max_features': ['sqrt', 'log2', None], 
        'max_depth': [3, 6, 9], 
        'max_leaf_nodes': [3, 6, 9], 
        } 
        return model, param