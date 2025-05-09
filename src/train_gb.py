import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from preprocess import preprocess_data


train_df = pd.read_csv('data/train.csv')


X_train, y_train, preprocessor = preprocess_data(train_df, is_train=True)


param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}


gb_model = GradientBoostingRegressor(random_state=42)


grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
grid_search.fit(X_train, y_train)


model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation MSE: {-grid_search.best_score_:.4f}")


with open('models/gb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("Gradient Boosting model (all features, optimized parameters) and preprocessor saved successfully!")  