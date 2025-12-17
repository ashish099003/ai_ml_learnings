import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def standardization(data):
    scaler_std = StandardScaler()
    data['Salary_Standardized'] = scaler_std.fit_transform(data[['Salary']])

    print("\nStandardized Data:\n", data[['Salary', 'Salary_Standardized']])
    return data


def min_max_normalization(data):
    min_max_scaler = MinMaxScaler()
    data['Salary_MinMax'] = min_max_scaler.fit_transform(data[['Salary']])
    print("\nNormalized Data:\n", data[['Salary', 'Salary_MinMax']])
    return data


if __name__ == '__main__':
    data = pd.DataFrame({
        'Salary': [30000, 45000, 55000, 60000, 75000, 120000]
    })
    print("Original Data:\n", data)
    data = standardization(data)
    data = min_max_normalization(data)
    print(data)
