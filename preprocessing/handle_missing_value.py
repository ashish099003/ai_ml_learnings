import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def imputation_using_mean(data):
    data_df = pd.DataFrame(data)
    mean_income = data_df['income'].mean()
    data_df['income'].fillna(mean_income, inplace=True)
    return data_df

def group_by_handle(data_df):
    return data_df.groupby('Occupation')['Income'].transform(lambda x: x.fillna(x.mean()))
def handle_missing_value():
    data = [
        {'age': 25, 'income': 50000, 'buy': 1},
        {'age': 40, 'income': None, 'buy': 0},
        {'age': 30, 'income': 70000, 'buy': 1},
    ]
    print(data)
    data = imputation_using_mean(data)
    print(data)

    # Sample customer data with missing incomes
    data = {
        'Age': [25, 30, 45, 40, 55, 50],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'Occupation': ['Engineer', 'Doctor', 'Doctor', 'Engineer', 'Lawyer', 'Lawyer'],
        'Income': [50000, np.nan, 80000, np.nan, 120000, 100000]
    }

    df = pd.DataFrame(data)
    print("Original Data:\n")
    print(df)
    df['Income'] = group_by_handle(df)
    print(df)

if __name__=='__main__':
    handle_missing_value()