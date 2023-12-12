!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv




import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sklearn.feature_extraction.text as text
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import xgboost
from sklearn import decomposition, ensemble
import numpy, textblob, string
import os
import re
import nltk
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

f1 = pd.read_csv('/content/data/full_dataset/goemotions_1.csv')
f2 = pd.read_csv('/content/data/full_dataset/goemotions_2.csv')
f3 = pd.read_csv('/content/data/full_dataset/goemotions_3.csv')

#data = pd.concat([f1, f2, f3], ignore_index= True)
data = pd.concat([f1], ignore_index= True)
#data.columns = ['text', '']
data.head()

# Assuming 'df' is the DataFrame from the previous code
# Melt the DataFrame to combine the emotion columns into a single 'emotion' column
melted = data.melt(id_vars=['text', 'example_very_unclear'], value_vars=data.columns[10:], var_name='emotion', value_name='emotion_value')

# Filtering to get rows where emotion_value is 1
melted = melted[melted['emotion_value'] == 1]

# Selecting only relevant columns
result = melted[['text', 'emotion']]

# Displaying the resulting DataFrame
print(result)

result.columns = ['text', 'emotion']
print(result)

# Dictionary mapping emotion column names to emotion names
emotion_mapping = {
    'admiration': 'admiration',
    'amusement': 'amusement',
    'anger': 'anger',
    'annoyance': 'annoyance',
    'approval': 'approval',
    'caring': 'caring',
    'confusion': 'confusion',
    'curiosity': 'curiosity',
    'desire': 'desire',
    'disappointment': 'disappointment',
    'disapproval': 'disapproval'
}

# Mapping the binary values to emotion names
result['emotion'] = result['emotion'].map(emotion_mapping)


# Reassigning result to our main dataframe
data = result

# Displaying the DataFrame with emotion names
print(data)

data.head()


nltk.download('stopwords')

# Converting uppercase to lowercase

data['text'] = data['text'].apply(lambda a: " ".join(a.lower() for a in a.split()))

# Removing whitespace and special characters

data['text'] = data['text'].apply(lambda a: " ".join(a.replace('[^\w\s]','') for a in a.split()))

"""***What are stopwords and why remove them?***

*Stopwords are common words like "the," "and," "of," etc., which occur frequently in language but often carry little specific meaning in a sentence. Removing them from text analysis helps focus on more important, content-bearing words, streamlining the process by filtering out ubiquitous but less informative terms. This aids in better identifying the core, meaningful words for tasks like sentiment analysis, text classification, or information retrieval, enhancing the accuracy and relevance of the analysis.*
"""

# Removing stopwords

stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda a: " ".join(a for a in a.split() if a not in stop))


spell = SpellChecker()

def correct_spellings(text):
    corrected_text = []
    words = text.split()
    misspelled_words = spell.unknown(words)
    word_correction_mapping = {word: spell.correction(word) if spell.correction(word) is not None else word for word in misspelled_words}

    for word in words:
        if word in word_correction_mapping:
            corrected_text.append(word_correction_mapping[word])
        else:
            corrected_text.append(word)

    return " ".join(corrected_text)

data['text'] = data['text'].apply(lambda a: correct_spellings(a))



# Normalizing

stem = PorterStemmer()
data['text'] = data['text'].apply(lambda a: " ".join([stem.stem(word) for word in a.split()]))


data['emotion'].value_counts()

# Transforming emotion categories to numerical categories

labelE = preprocessing.LabelEncoder()
data['emotion'] = labelE.fit_transform(data['emotion'])

data['emotion'].value_counts()

# Checking data after preprocessing

data.head()

# Train and Test Split

Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(data['text'], data['emotion'],stratify= data['emotion'])


# Instantiate the CountVectorizer
countV = CountVectorizer()

# Fit the CountVectorizer with the 'text' data from the entire dataset
countV.fit(data['text'])

# Transform the text data of the training set (Xtrain) into a document-term matrix
cv_xtrain = countV.transform(Xtrain)

# Transform the text data of the testing set (Xtest) into a document-term matrix using the same CountVectorizer
cv_xtest = countV.transform(Xtest)

# Create a TF-IDF Vectorizer instance
tVect = TfidfVectorizer()

# Fit the TF-IDF Vectorizer with the 'text' data from the entire dataset
tVect.fit(data['text'])

# Transform the text data of the training set (Xtrain) using the TF-IDF Vectorizer
tv_xtrain = tVect.transform(Xtrain)

# Transform the text data of the testing set (Xtest) using the same TF-IDF Vectorizer
tv_xtest = tVect.transform(Xtest)

def build(model, X_train, target, X_test):
  # Train the model
  model.fit(X_train, target)

  # Predict using the trained model
  predictions = model.predict(X_test)

  # Calculate and return accuracy
  return metrics.accuracy_score(predictions, Ytest)


# Naive Bayes Model with count vectors

cv_NBresult = build(naive_bayes.MultinomialNB(), cv_xtrain, Ytrain, cv_xtest)

print(cv_NBresult)

# Naive Bayes Model with count vectors

tv_NBresult = build(naive_bayes.MultinomialNB(), tv_xtrain, Ytrain, tv_xtest)

print(tv_NBresult)


cv_RFresult = build(ensemble.RandomForestClassifier(), cv_xtrain, Ytrain, cv_xtest)

print(cv_RFresult)

tv_RFresult = build(ensemble.RandomForestClassifier(), tv_xtrain, Ytrain, tv_xtest)

print(tv_RFresult)

# Confusion Matrix

classifier = linear_model.LogisticRegression().fit(tv_xtrain, Ytrain)
val_predictions = classifier.predict(tv_xtest)

# Precision , Recall , F1 - score , Support
y_true, y_pred = Ytest, val_predictions
print(classification_report(y_true, y_pred))
print()

# Connecting to Twitter API

import requests
import pandas as pd

tData = []

payload = {
    'api_key':'ENTER YOUR SCRAPER API KEY',
    'query':'Meta',
    'num':'500'

}

res = requests.get(
    'https://api.scraperapi.com/structured/twitter/search',params = payload
)

data = res.json()

data.keys()

allTweets = data['organic_results']
for tweet in allTweets:
  tData.append(tweet)

df = pd.DataFrame(tData)
df.to_json('tweets.json', orient = 'index')
print("exported")

twt = pd.read_json('tweets.json', lines = True, orient = 'records')

twt = twt.to_csv('twt.csv', index = False)

twt = pd.read_csv('twt.csv')

twt = df[['snippet']]

twt.tail()

Xpredict = twt['snippet']

pred_tfidf = tVect.transform(Xpredict)
twt['Emotion'] = classifier.predict(pred_tfidf)
twt.tail() #Change twt

twt['sentiment'] = twt['snippet'].apply(lambda a: TextBlob(a).sentiment[0] )
def function (value):
     if value['sentiment'] < 0 :
        return 'Negative'
     if value['sentiment'] > 0 :
        return 'Positive'
     return 'Neutral'

twt['Sentiment_label'] = twt.apply (lambda a: function(a),axis=1)
twt.tail()

! pip install chart_studio

import chart_studio.plotly as py
import plotly as ply
import cufflinks as cf
from plotly.graph_objs import *
from plotly.offline import *
from IPython.display import display, HTML

init_notebook_mode(connected=True)
cf.set_config_file(offline=True, world_readable=True, theme='white')

Sentiment_df = pd.DataFrame(twt.Sentiment_label.value_counts().reset_index())
Sentiment_df.columns = ['sentiment', 'Count']
Sentiment_df = pd.DataFrame(Sentiment_df)
Sentiment_df['Percentage'] = 100 * Sentiment_df['Count']/ Sentiment_df['Count'].sum()
Sentiment_Max = Sentiment_df.iloc[0,0]


Sentiment_percent = str(round(Sentiment_df.iloc[0,2],2))
fig1 = Sentiment_df.iplot(kind='pie',labels='sentiment',values='Count',textinfo='label+percent', title= 'Sentiment Analysis', world_readable=True,
                    asFigure=True)
ply.offline.plot(fig1,filename="sentiment")

# Use IPython's display() function to read and display the HTML file
display(HTML(filename='Sentiment.html'))

import chart_studio.plotly as py
import plotly as ply
import cufflinks as cf
from plotly.graph_objs import *
from plotly.offline import *
from IPython.display import display, HTML

init_notebook_mode(connected=True)
cf.set_config_file(offline=True, world_readable=True, theme='white')
Emotion_df = pd.DataFrame(twt.Emotion.value_counts().reset_index())
Emotion_df.columns = ['Emotion', 'Count']
Emotion_df = pd.DataFrame (Emotion_df)

# Convert 'Emotion' column to string type
#Emotion_df['Emotion'] = Emotion_df['Emotion'].astype(str)

Emotion_df['Percentage'] = 100 * Emotion_df['Count']/ Emotion_df['Count'].sum()
Emotion_Max = Emotion_df.iloc[0,0]
Emotion_percent = str(round(Emotion_df.iloc[0,2],2))
fig = Emotion_df.iplot(kind='pie', labels = 'Emotion', values = 'Count',pull= .2, hole=.2 , colorscale = 'reds', textposition='outside',colors=['red','green','purple','orange','blue','yellow','pink'],textinfo='label+percent', title= 'Emotion Analysis', world_readable=True,asFigure=True)
ply.offline.plot(fig,filename="Emotion")


# Use IPython's display() function to read and display the HTML file
display(HTML(filename='Emotion.html'))

