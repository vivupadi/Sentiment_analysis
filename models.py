from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

class models:
    def __init__(self):
        pass

    def log_reg(self):
        return LogisticRegression(C=1, solver = 'liblinear',max_iter=150)
    
    def XGB(self):
        return XGBClassifier()
    
    def RanFo(self):
        return RandomForestClassifier()