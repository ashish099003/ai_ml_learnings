
import pandas as pd
def one_hot_enc(x):
    '''x is a dictionary
    Output -> A dataframe with required modifications should be returned'''

    x = pd.DataFrame(x, columns = ['towns', 'area', 'Price'])

    # to_rem list consists of the name of all the columns of x
    to_rem = list(x.columns)

    # Iterate through all the unique values in the columns 'towns'
    for i in x['towns'].unique():
        # Create new columns with name of towns with prefix "town" and equate the values to zero
        x["town_{}".format(i)] = 0

        # Equate the values of particular columns of rows of certain 'town'
        x.loc[x['towns' ]==i, 'town_{}'.format(i)] = 1

    lst = list(x.columns)

    # new_col will have the names of all columns which are present in x but not in to_rem
    new_col = [elem for elem in lst if elem not in to_rem]

    # return a dataframe with sorted town-prefix columns and area, price in the end.
    return x[sorted(new_col) + ['area' ,'Price']]


if __name__=='__main__':
    x = {'towns': ['Monroe', 'Monroe', 'Monroe', 'Monroe', 'Windsor', 'Windsor', 'Windsor','Robinsville', 'Robinsville', 'Robinsville'], 'area': [2600, 3000, 3200, 3600, 2600, 2800, 3300, 2600, 2900, 3600], 'Price': [550000, 565000, 610000, 680000, 725000, 585000, 615000, 710000, 620000, 695000]}
    print(one_hot_enc(x))




