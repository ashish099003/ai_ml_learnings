import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def impute_mean_missing_data(df):
    income_mean = df['Income'].mean()
    df['Income'].fillna(income_mean,inplace=True)
    return df

def impute_median(df):
    meadian_income = df['Income'].median()
    df['Income'].fillna(meadian_income,inplace=True)
    print(meadian_income)
    return df


if __name__=='__main__':

    data = {
        'Age': [25, 30, 45, 40, 55, 50],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'Occupation': ['Engineer', 'Doctor', 'Doctor', 'Engineer', 'Lawyer', 'Lawyer'],
        'Income': [50000, np.nan, 80000, np.nan, 120000, 100000]
    }

    df = pd.DataFrame(data)
    # print("Original Data:\n")
    print(df)
    # df_mean = impute_mean_missing_data(df)
    # print("After filling missing value with mean: \n")
    # print(df_mean)
    df_median = impute_median(df)
    print("After filling missing value with median: \n")
    print(df_median)

