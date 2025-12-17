from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import pandas as pd


def label_encoding():
    data = pd.DataFrame({
        'Region': ['North America', 'Europe', 'Asia', 'Australia', 'Europe', 'Asia']
    })
    label_encoder = LabelEncoder()
    data['labeled_Region'] = label_encoder.fit_transform(data['Region'])
    print(data)

def ordinal_encoding():
    # Sample dataset with ordered categories (Education levels)
    data = pd.DataFrame({
        'Education_Level': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master']
    })
    print("Original Data:\n", data)
    categories = ['High School', 'Bachelor', 'Master', 'PhD']
    ordinal_encoder = OrdinalEncoder(categories=[categories])
    data['ordinal_Education_Level'] = ordinal_encoder.fit_transform(data[['Education_Level']])
    print(data)

def one_hot_encoding():
    data = pd.DataFrame({'City': ['New York', 'Paris', 'London', 'Paris']})
    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(data[['City']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['City']))
    print(encoded_df)

def dummies():
    data = pd.DataFrame({'City': ['New York', 'Paris', 'London', 'Paris']})
    dummies_df = pd.get_dummies(data, columns=['City'])
    print(dummies_df)

if __name__=='__main__':
    # label_encoding()
    # ordinal_encoding()
    # one_hot_encoding()
    dummies()