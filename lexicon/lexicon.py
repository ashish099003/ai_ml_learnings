
import numpy as np
import pandas as pd
from numpy.linalg import norm
import warnings
from nrclex import NRCLex
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from afinn import Afinn
from textblob import TextBlob
import spacy
#Importing necessary libraries
import gensim
import gensim.corpora as corpora
nltk.download('stopwords')
warnings.filterwarnings('ignore')

def cosineSimilarity(A, B):
    """
    Write your code here to calculate two user input in list
    """
    array_a = np.array(A)
    array_b = np.array(B)
    norm_a = norm(array_a)
    norm_b = norm(array_b)
    cosSim = (np.dot(array_a,array_b))/(norm_a*norm_b)

    return np.round(cosSim ,3)

def text_blob():
    # text = "The service was outstanding and the food was delicious!"
    text = "The service was  bad"
    blob = TextBlob(text)
    sentiment = blob.sentiment
    print("Input Text Sentiment: ", sentiment.subjectivity)
    print("Input Text Polarity: ", sentiment.polarity)
    text2 = "Learning AI ML is interesting but this require punctuality and not good for career"
    blob_2 = TextBlob(text2)
    print(f"Polarity of text : {text2} is {blob_2.polarity}")
    print(f"Sentiment for  text : {text2} is {blob_2.subjectivity}")

def train_csv_sentiment():
    training = pd.read_csv("train.csv")
    print(training.head())
    print("Information of training dataset:\n", training.info())
    neg = training["negatives"]
    pos = training["positives"]
    print((pos.iloc[0]))
    blob2 = TextBlob(pos.iloc[0])
    print("Polarity:", blob2.sentiment.polarity)
    print("Subjectivity:", blob2.sentiment.subjectivity)
    print(neg.iloc[0])
    blob2 = TextBlob(neg.iloc[0])
    print("Polarity:", blob2.sentiment.polarity)
    print("Subjectivity:", blob2.sentiment.subjectivity)

    # Performing sentiment analysis on the negative reviews
    slice_neg = neg[4:8, ]
    # Executing the for loop for displaying multiple reviews
    for review in slice_neg:
        print('Review is:', review)
        print('Predicted Polarity:', TextBlob(review).sentiment.polarity)
        print('.' * 50)
    # Performing sentiment analysis on the poistive reviews
    slice_pos = pos[14:18, ]

    for review in slice_pos:
        print('Review is:', review)
        print('Predicted Polarity:', TextBlob(review).sentiment.polarity)
        print('.' * 50)


def sentiment_ana_using_nrc():
    text = "I love the new phone design, but I'm scared about its durability."
    emotions = NRCLex(text)
    # NRCLex(text): This breaks your input text into words and checks each one against the NRC lexicon.

    # Display raw emotion scores
    print("Raw Emotion Scores:", emotions.raw_emotion_scores)
    # .raw_emotion_scores: Returns a dictionary with the count of words associated with each emotion.

    # Display top emotions in the sentence
    print("Top Emotions:", emotions.top_emotions)
    # .top_emotions: Gives the most frequent emotions found in the text

    text = "Although I am worried, I still trust the process and feel excited."
    print("*"*50, "\n")
    emotion = NRCLex(text)

    print("Words in Lexicon:", emotion.words)
    print("Raw Emotion Scores:", emotion.raw_emotion_scores)
    print("Top Emotions:", emotions.top_emotions )
    print("Emotion Frequencies:", emotion.affect_frequencies )


    print("\n", "*"*50, "\n")

def word_with_emotions():
    # 📝 Define a sample sentence
    text = "The movie made me happy, anxious and full of anticipation."

    # 🧪 Step 1: Analyze text
    emotion = NRCLex(text)

    # 📤 Step 2: Get the raw emotion mapping
    raw_dict = emotion.raw_emotion_scores

    # 🧱 Step 3: Split the text into words
    words = text.lower().split()

    # 📦 Step 4: Create a DataFrame of words and their emotions
    emotion_data = []

    for word in words:
        word_emotion = NRCLex(word)
        for emo in word_emotion.raw_emotion_scores.keys():
            emotion_data.append({'Word': word, 'Emotion': emo, 'Flag': 1})

    # Convert to DataFrame
    df = pd.DataFrame(emotion_data)

    # 🧊 Step 5: Create Pivot Table
    pivot_table = pd.pivot_table(df, values='Flag', index='Word', columns='Emotion', fill_value=0)

    # 🖨️ Display the Pivot Table
    print("🔁 Pivot Table of Words vs Emotions:")
    print(pivot_table)
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', cbar=False)
    plt.title('Emotion Mapping of Words (NRC Lexicon)')
    plt.show()

def sentiment_using_af():

    af = Afinn()
    # 📝 Define input sentences
    sentences = [
        "The service was amazing and the staff was wonderful!",
        "The food was horrible and I hated the experience.",
        "The hotel is okay, nothing special.",
        "Absolutely love this place!",
        "The customer support is terrible!"
    ]

    # 🧪 Analyze each sentence
    for sentence in sentences:
        score = af.score(sentence)
        print(f"'{sentence}' --> Sentiment Score: {score}")
    
    print("\n", "*"*50, "\n")

    for review in sentences:
        print("Words in the review:", af.split(review))
        print("Emotions in the review:", af.find_all(review))
        print("Score of each emotion:", af.scores_with_pattern(review))
        print("Total score of emotions:", af.score_with_wordlist(review))
        print("\n")

    print("\n", "*"*50, "\n")
    review2 = 'Great, superb,excellent, wonderful, beautiful and amazing place!'
    print("Words in the review:", af.split(review2))
    print("Emotions in the review:", af.find_all(review2))
    print("Score of each emotion:", af.scores_with_pattern(review2))
    print("Total score of emotions:", af.score_with_wordlist(review2))
    print("\n")

    review3 = 'Pathetic, disgusting and worse food'
    print("Words in the review:", af.split(review3))
    print("Emotions in the review:", af.find_all(review3))
    print("Score of each emotion:", af.scores_with_pattern(review3))
    print("Total score of emotions:", af.score_with_wordlist(review3))

if __name__=='__main__':
    # A = [2,1,2,3,2,9]
    # B = [3,4,2,4,5,5]
    # print(cosineSimilarity(A, B))
    # text_blob()
    # train_csv_sentiment()
    # sentiment_ana_using_nrc()
    # word_with_emotions()
    sentiment_using_af()