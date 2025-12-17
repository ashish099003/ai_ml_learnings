import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

def predict_with_logistic():
    # Dataset
    X = np.array([[30], [35], [40], [45], [50], [60]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_lin_pred = lin_reg.predict(X)

    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    y_log_pred = log_reg.predict_proba(X)[:,1]

    # Plotting
    plt.figure(figsize=(10, 5))

    # Linear
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X, y_lin_pred, color='green', label='Linear Prediction')
    plt.title("Linear Regression Fit")
    plt.xlabel("Income (₹1000s)")
    plt.ylabel("Purchased")
    plt.ylim(-0.5, 1.5)
    plt.legend()

    # Logistic
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X, y_log_pred, color='red', label='Logistic Curve')
    plt.title("Logistic Regression Curve")
    plt.xlabel("Income (₹1000s)")
    plt.ylabel("Probability of Purchase")
    plt.ylim(-0.1, 1.1)
    plt.legend()

    plt.tight_layout()
    plt.show()

def sigmoid_curve():
    z = np.linspace(-10,10,100)

    sigmoid = 1/(1+ np.exp(-z))
    plt.figure(figsize=(6, 4))
    plt.plot(z, sigmoid, color='purple')
    plt.title("Sigmoid Function")
    plt.xlabel("z (Weighted Sum)")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.6)
    plt.axvline(0, linestyle='--', color='gray', alpha=0.6)
    plt.show()

def probability_of_eradictating_tumor():
    w0 = -6
    w1 = 0.05
    w2 = 1
    z = w0 + w2*3.5 + w1*50
    prob = 1/(1 + np.exp(-z))
    print(prob)


def log_loss():
    z = [3.11, 0.08, 0.76, 5.98, 3.05, 0.12, 8.99, 1.69, 1.75, 1.54]
    y_true = [1, 0, 1, 0, 0, 0, 1, 1, 0, 1]
    x = [[11, 22], [39, 0], [33, 39], [1, 28], [9, 24], [19, 14], [6, 7], [28, 3], [4, 17], [35, 15]]
    z = np.asarray(z)
    y_true = np.asarray(y_true)
    x = np.asarray(x)
    y_pred = 1 / (1 + np.exp(-z))
    print(y_pred)
    log_losss = -y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)
    print(np.round(log_losss,2))
    log_derv = []
    for i in range(0,len(y_true)):
        log_derv.append((y_pred[i]-y_true[i])*(x[i][0]))
    print(np.round(log_derv,2))


if __name__=='__main__':
    # predict_with_logistic()
    # probability_of_eradictating_tumor()
    # log_loss()
    sigmoid_curve()