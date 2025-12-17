import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def treat_outliers():

    data = np.array([10, 12, 13, 14, 15, 16, 18, 50, 52, 550])
    df = pd.DataFrame({'Value': data})
    q1 = df['Value'].quantile(0.25)
    q3 = df['Value'].quantile(0.75)
    iqr = q3 - q1
    lower_end = q1 - 1.5*iqr
    high_end = q3 + 1.5*iqr
    df['Value_Capped'] = df['Value'].clip(lower=lower_end, upper=high_end)
    df['Value_logged'] = np.log(df['Value_Capped'])
    plt.figure(figsize=(15, 4))
    # Original data
    plt.subplot(1, 3, 1)
    sns.boxplot(y=df['Value'], color='skyblue')
    plt.title('Original Data')

    # Capped data
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df['Value_Capped'], color='lightgreen')
    plt.title('Capped Data')

    # Log-transformed data
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df['Value_logged'], color='lightcoral')
    plt.title('Log-transformed Data')

    plt.tight_layout()
    plt.show()



if __name__=='__main__':
    treat_outliers()