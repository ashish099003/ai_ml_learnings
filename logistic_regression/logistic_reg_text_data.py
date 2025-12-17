import pandas as pd
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords (run only once)
nltk.download('stopwords')
nltk.download('punkt')  # Needed for word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report, cohen_kappa_score, f1_score, recall_score

# Prepare stopwords list
stop_words = set(stopwords.words('english'))


def data_understand():
    data = pd.read_csv('/Users/ashish/PycharmProjects/pythonTest/test/logistic_regression/tripadvisor_hotel_reviews.csv')
    print("Information of dataset:\n")
    # print(data.info())
    # print(data.head())

    num_categories = data['Rating'].nunique()
    print(f"\n number of unique categories: {num_categories}")
    categories_list = data['Rating'].unique()
    print(f"\n List of Categories: {categories_list}")

    missing_value = data.isnull().sum()
    print(f"\n Missing value in data : {missing_value}")

    X = data['Review']
    Y = data['Rating']
    return X,Y

def preprocessing(X):
    review_list = []
    def process_review(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = text.strip()
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(filtered_tokens)

    for review in X:
        processed = process_review(review)
        review_list.append(processed)

    cv = CountVectorizer(binary=False, min_df=2, ngram_range=(1,2))
    cv_result= cv.fit_transform(review_list)
    print("Dimension after CountVectorizer:", cv_result.shape)
    return cv_result

def train_model_evaluate(cv_result, Y):
     X_train, X_test, y_train, y_test = train_test_split(cv_result, Y, test_size=0.2, stratify=Y)
     lr = LogisticRegression()
     lr.fit(X_train,y_train)
     y_pred = lr.predict(X_test)
     y_train_pred = lr.predict(X_train)

     train_accuracy = accuracy_score(y_train,y_train_pred)
     test_accuracy = accuracy_score(y_test,y_pred)
     print("Training Accuracy:", train_accuracy)
     print("Test Accuracy:", test_accuracy)
     print("ConfusionMatrix : ", confusion_matrix(y_test,y_pred))
     print("Classification Report: ", classification_report(y_test,y_pred))
     print("R2 Score:", r2_score(y_test, y_pred))
     print("Cohen's Kappa:", cohen_kappa_score(y_test, y_pred))
     print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
     print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))


if __name__=='__main__':
    X,Y = data_understand()
    cv_result = preprocessing(X)
    train_model_evaluate(cv_result,Y)
