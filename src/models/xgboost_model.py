import xgboost as xgb
import numpy as np

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3
        )
        self.is_trained = False

    def predict_confidence(self, state):
        if not self.is_trained:
            # Return uniform probability if not yet trained
            return np.array([0.33, 0.33, 0.34])
        
        # state expected as (batch, features)
        return self.model.predict_proba(state)

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
