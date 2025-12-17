import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import  Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler , PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


def cars_24_lr():
    df = pd.read_csv('cars24-car-price-cleaned-new.csv')
    # print(df.head())
    # print(df['model'].nunique(), df['make'].nunique())
    # df['make']= df.groupby('make')['selling_price'].mean()
    # df['model'] = df.groupby('model')['selling_price'].mean()
    #
    # # print(df.head())
    # scalar = MinMaxScaler()
    # df = pd.DataFrame(scalar.fit_transform(df), columns=df.columns)
    # print(df.head())

    df_train, df_test = train_test_split(df, test_size=0.3)
    # print(df_train)
    # print(df_test)
    make_category_wise_mean = df_train.groupby('make')['selling_price'].mean()
    model_category_wise_mean = df_train.groupby('model')['selling_price'].mean()
    df_train['make'] = df_train.groupby('make')['selling_price'].transform('mean')
    df_train['model'] = df_train.groupby('model')['selling_price'].transform('mean')


    df_test['make'] = df_test['make'].map(make_category_wise_mean)
    df_test['model'] = df_test['model'].map(model_category_wise_mean)
    # print(df_test.isna().sum())
    global_mean = df_train['selling_price'].mean()
    df_test['make'] = df_test['make'].fillna(global_mean)
    df_test['model'] = df_test['model'].fillna(global_mean)
    # print(df_train.head())
    scaler = MinMaxScaler()
    df_train= pd.DataFrame(scaler.fit_transform(df_train), columns=df.columns)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df.columns)
    # print(df_train.head())
    y_train = df_train['selling_price']
    x_train = df_train.drop('selling_price', axis=1)
    # print(y_train)
    # print(x_train)
    y_test = df_test['selling_price']
    X_test = df_test.drop('selling_price', axis=1)
    model = LinearRegression()
    model.fit(x_train,y_train)
    print(model.coef_)
    print(model.score(x_train, y_train))
    y_pred_train = model.predict(x_train)
    residuals_train = y_train - y_pred_train
    # plt.hist(residuals_train, bins=30)
    # plt.show()

    # #predicting with one feature model
    #
    # y_hat = model.predict(X_test[['model']])
    # plt.scatter(df_test['model'], df_test['selling_price'], label='data')
    # plt.show()
    #
    # feature_importance = pd.Series(abs(model.coef_), index=x_train.columns)
    # feature_importance.sort_values(ascending=False,inplace=True)
    # most_important_feature = feature_importance.index[0]
    # print(f"The most important feature is: {most_important_feature}")
    #
    # # Plot feature importance
    # plt.figure(figsize=(10, 6))
    # feature_importance.plot(kind='bar')
    # plt.title('Feature Importance')
    # plt.ylabel('Absolute Coefficient Value')
    # plt.show()
    #
    #
    # #calculate adjusted R2
    # n = x_train.shape[0]
    # p = x_train.shape[1]
    # r2 = model.score(x_train,y_train)
    # adjusted_r2 =  1 - (1 - r2) * (n - 1) / (n - p - 1)
    # print(f"Adjusted R-squared: {adjusted_r2}")

    # calculate VIF
    vif = pd.DataFrame()
    vif['feature'] = x_train.columns
    vif['vif'] = [variance_inflation_factor(x_train.values,i) for i in range(x_train.shape[1])]
    # print(vif)
    vif = vif.sort_values(by='vif',ascending=False)
    feature_to_remove = vif.iloc[0]['feature']
    print(feature_to_remove)

    x_train = x_train.drop(columns=[feature_to_remove])
    X_test = X_test.drop(columns=[feature_to_remove])
    model_new = LinearRegression()
    model_new.fit(x_train,y_train)

    # # Recalculate VIF for the remaining features
    # vif_data_new = pd.DataFrame()
    # vif_data_new["feature"] = x_train.columns
    # vif_data_new["VIF"] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
    #
    # # Display the new VIF values
    # print(vif_data_new)

    # Check for underfitting/overfitting
    # model.fit(x_train, y_train)
    # train_score = model.score(x_train, y_train)
    # test_score = model.score(X_test, y_test)
    #
    # print(f"Train R-squared: {train_score}")
    # print(f"Test R-squared: {test_score}")
    #
    # if train_score > test_score + 0.05:  # A common heuristic for overfitting
    #     print("The model might be overfitting.")
    # elif test_score > train_score + 0.05:  # A common heuristic for underfitting
    #     print("The model might be underfitting.")
    # else:
    #     print("The model is neither significantly underfitting nor overfitting.")


    # We will be trying now polynomial feature to enhance the score.

    degrees = [1,2,3,4]
    train_r2 = []
    test_r2 = []
    for degree in degrees:
        model_pipeline = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        model_pipeline.fit(x_train,y_train)
        print(degree, "->", model_pipeline.score(x_train, y_train), model_pipeline.score(X_test, y_test))
        train_r2.append(model_pipeline.score(x_train, y_train))
        test_r2.append(model_pipeline.score(X_test, y_test))
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(degrees, train_r2, marker='o', label='Train R-squared')
    # plt.plot(degrees, test_r2, marker='o', label='Test R-squared')
    # plt.title('Train vs Test R-squared for Different Polynomial Degrees')
    # plt.xlabel('Polynomial Degree')
    # plt.ylabel('R-squared')
    # plt.xticks(degrees)
    # plt.grid(True)
    # plt.legend()
    # plt.show()
   #
   # """
   # Lasso Regression && Ridge Regression
   # """
    alphas = [0.001, 0.01, 0.1, 1, 10]
    print("Lasso Regression:")
    print("=" * 30)
    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(x_train,y_train)
        train_r2_score = lasso_model.score(x_train,y_train)
        test_r2_score = lasso_model.score(X_test,y_test)
        print(f"Alpha: {alpha}, Train R-squared: {train_r2_score:.4f}, Test R-squared: {test_r2_score:.4f}")

    print("\n" + "=" * 30 + "\n")  # Separator

    print("Ridge Regression:")
    print("=" * 30)
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(x_train, y_train)
        train_r2_ridge = ridge_model.score(x_train, y_train)
        test_r2_ridge = ridge_model.score(X_test, y_test)
        print(f"Alpha: {alpha}, Train R-squared: {train_r2_ridge:.4f}, Test R-squared: {test_r2_ridge:.4f}")


if __name__=='__main__':
    cars_24_lr()