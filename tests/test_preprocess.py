import pandas as pd
from src.preprocess import preprocess_data

def test_preprocess():
    df = pd.DataFrame({'A': [1, 2, None], 'B': ['yes', 'no', 'yes']})
    processed_df = preprocess_data(df, is_train=False)
    assert processed_df.isnull().sum().sum() == 0  # Check if NaN exists ? 
