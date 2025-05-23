import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


ZERO_FILL_COLS = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 
                  'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
ORDINAL_COLS = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
                'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
NOMINAL_COLS = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 
                'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 
                'SaleCondition']

def load_data(train_path, test_path=None):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df

def preprocess_data(df, is_train=True, preprocessor=None, selected_features=None):
    
    df = df.drop('Id', axis=1, errors='ignore')
    
    
    if selected_features is not None:
        columns_to_keep = selected_features + ['SalePrice'] if is_train else selected_features
        df = df[columns_to_keep]
    
    
    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ZERO_FILL_COLS and col != 'SalePrice']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(q_low, q_high)
    
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')
    
    
    print("NaN values before transformation:")
    print(df.isna().sum()[df.isna().sum() > 0])
    
    if is_train:
        
        original_len = len(df)
        
        
        dummy_rows = []
        for col in ORDINAL_COLS:
            if col in df.columns and 'missing' not in df[col].values:
                dummy_row = pd.DataFrame({col: ['missing']}, index=[len(df) + len(dummy_rows)])
                for num_col in numeric_cols + ZERO_FILL_COLS:
                    if num_col in df.columns:
                        dummy_row[num_col] = df[num_col].median()
                for cat_col in categorical_cols:
                    if cat_col != col and cat_col in df.columns:
                        dummy_row[cat_col] = 'missing'
                dummy_row['SalePrice'] = np.nan
                dummy_rows.append(dummy_row)
        
        if dummy_rows:
            df = pd.concat([df] + dummy_rows, ignore_index=True)

        
        X = df.drop('SalePrice', axis=1, errors='ignore')
        y = df['SalePrice']
        
        
        available_numeric_cols = [col for col in (numeric_cols + ZERO_FILL_COLS) if col in X.columns]
        available_ordinal_cols = [col for col in ORDINAL_COLS if col in X.columns]
        available_nominal_cols = [col for col in NOMINAL_COLS if col in X.columns]
        
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), available_numeric_cols),
                ('ord', Pipeline([
                    ('label', LabelEncoderWrapper(available_ordinal_cols)),
                ]), available_ordinal_cols),
                ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_nominal_cols)
            ])
        
        
        X_processed = preprocessor.fit_transform(X)
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'ord':
                feature_names.extend(cols)
            elif name == 'nom':
                feature_names.extend(transformer.get_feature_names_out(cols))
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
        
        
        print("NaN values after transformation:")
        print(X_processed.isna().sum()[X_processed.isna().sum() > 0])
        
        
        if X_processed.isna().any().any():
            print("Warning: NaN values found after transformation. Imputing with 0.")
            X_processed = X_processed.fillna(0)
        
        
        X_processed = X_processed.iloc[:original_len]
        y = y.iloc[:original_len]
        
        
        print("NaN values in y:")
        print(y.isna().sum())
        
        
        if y.isna().any():
            print("Warning: NaN values found in y. Dropping rows with NaN in y.")
            valid_indices = y.dropna().index
            X_processed = X_processed.loc[valid_indices]
            y = y.loc[valid_indices]
        
        return X_processed, y, preprocessor
    else:
        
        X_processed = preprocessor.transform(df)
        
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'ord':
                feature_names.extend(cols)
            elif name == 'nom':
                feature_names.extend(transformer.get_feature_names_out(cols))
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
        
        
        print("NaN values after transformation (test):")
        print(X_processed.isna().sum()[X_processed.isna().sum() > 0])
        
        
        if X_processed.isna().any().any():
            print("Warning: NaN values found after transformation (test). Imputing with 0.")
            X_processed = X_processed.fillna(0)
        
        return X_processed

def preprocess_user_input(user_input, feature_cols, preprocessor, train_df):
    input_df = pd.DataFrame([user_input], columns=feature_cols)
    
    
    for col in ZERO_FILL_COLS:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna(0)
    
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ZERO_FILL_COLS and col != 'SalePrice' and col != 'Id']
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna(train_df[col].median())
            
            q_low = train_df[col].quantile(0.01)
            q_high = train_df[col].quantile(0.99)
            input_df[col] = input_df[col].clip(q_low, q_high)
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna('missing')
    
    
    input_processed = preprocessor.transform(input_df)
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'ord':
            feature_names.extend(cols)
        elif name == 'nom':
            feature_names.extend(transformer.get_feature_names_out(cols))
    input_processed = pd.DataFrame(input_processed, columns=feature_names)
    
    
    print("Processed input for prediction:")
    print(input_processed)
    
    
    if input_processed.isna().any().any():
        print("Warning: NaN values found in user input after transformation. Imputing with 0.")
        input_processed = input_processed.fillna(0)
    
    return input_processed

class LabelEncoderWrapper:
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {col: LabelEncoder() for col in columns}
        self.fitted = False
    
    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                self.encoders[col].fit(X[col])
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("LabelEncoderWrapper must be fitted before calling transform")
        X_copy = X.copy()
        for col in self.columns:
            if col in X.columns:
                encoder = self.encoders[col]
                classes = encoder.classes_
                X_copy[col] = X_copy[col].apply(lambda x: x if x in classes else classes[0])
                X_copy[col] = encoder.transform(X_copy[col])
        return X_copy
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)