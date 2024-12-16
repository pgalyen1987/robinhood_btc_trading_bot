import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from utils.logger import logger

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, profit_target: float = 0.0005):
        self.df = df
        self.profit_target = profit_target
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        try:
            # Create target variable
            self.df['Target'] = (self.df['Close'].shift(-1) > 
                               self.df['Close'] * (1 + self.profit_target)).astype(int)
            
            feature_columns = ['SMA20', 'EMA20', 'RSI', 'BB_upper', 'BB_lower', 
                             'Price_Change', 'Price_Change_Lag1', 'Volatility']
            
            X = self.df[feature_columns]
            y = self.df['Target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_model(self):
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data()
        self.model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(self.model, 'ml_module/models/rf_model.joblib')
        joblib.dump(self.scaler, 'ml_module/models/scaler.joblib')
        
        return self.model.score(X_test_scaled, y_test) 