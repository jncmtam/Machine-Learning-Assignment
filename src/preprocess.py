import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(train_path, test_path=None): 
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df

def preprocess_data(df, is_train=True):
    df = df.fillna(df.mean(numeric_only=True))
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    if is_train:
        X = df.drop('SalePrice', axis=1)
        y = df['SalePrice']
        return X, y
    return df

def preprocess_user_input(user_input, feature_cols, encoders, train_df):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input], columns=feature_cols)
    
    # Exclude 'Id' and 'SalePrice' from numeric columns since 'Id' is not a feature and 'SalePrice' is the target
    numeric_cols = train_df.select_dtypes(exclude=['object']).columns.drop(['Id', 'SalePrice'])
    numeric_means = train_df[numeric_cols].mean()
    input_df[numeric_cols] = input_df[numeric_cols].fillna(numeric_means)
    
    # Handle categorical columns: only encode provided values
    for col, encoder in encoders.items():
        if col in input_df.columns and pd.notna(input_df[col].iloc[0]):
            try:
                input_df[col] = encoder.transform([input_df[col].iloc[0]])[0]
            except ValueError as e:
                print(f"Error: Invalid value for {col}: {input_df[col].iloc[0]}. Using default encoding.")
                input_df[col] = encoder.transform([train_df[col].mode()[0]])[0]  # Use mode as fallback
    
    # Fill remaining NaN with 0 (for numeric columns not in user_input)
    input_df = input_df.fillna(0)
    return input_df

def get_encoders(train_df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        encoder.fit(train_df[col].fillna('missing'))  # Handle NaN in training data
        encoders[col] = encoder
    return encoders