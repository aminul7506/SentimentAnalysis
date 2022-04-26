import pandas as pd
import re
import nltk
import tensorflow

from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from autocorrect import Speller
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

# choose classifier
print("Choose your classifier.")
print("Press 1 for GaussianNB.")
print("Press 2 for LinearRegression")
print("Press 3 for LogisticRegression.")
print("Press 4 for Linear Support Vector Machine.")
print("Press 5 for Polynomial Support Vector Machine with degree 2.")
print("Press 6 for Polynomial Support Vector Machine with degree 3.")
classifierValue = input("Enter your value: ")

# choose feature extraction method
print("Choose feature extraction method.")
print("Press 1 for TF-IDF vectorizer.")
print("Press 2 for Count vectorizer.")
vectorizerValue = input("Enter your value: ")

# read columns of text and sentiment of movie reviews dataset
columns = ["sentiment", "review"]
review_df = pd.read_csv("../Data/IMDB_Dataset_small.csv", usecols=columns)
reviews = review_df.review.values
sentiments_initial = review_df.sentiment.values

# download nltk words and stop words
nltk.download('punkt')
nltk.download('stopwords')

# pre-processing
spell = Speller(lang='en')
stemmer = PorterStemmer()
data = []

for i in range(len(reviews)):
    review = reviews[i]
    # remove non alphabetic characters
    review = re.sub('[^A-Za-z]', ' ', review)
    # make words lowercase
    review = review.lower()
    # tokenizing
    tokenized_tweet = wt(review)
    # remove stop words and stemming
    processed_review_initial = []
    for word in tokenized_tweet:
        if word not in set(stopwords.words('english')):
            stemmed_word = stemmer.stem(word)
            spelled_correction_with_stemmed_word = spell(stemmed_word)
            processed_review_initial.append(spelled_correction_with_stemmed_word)

    # join words to form sentence again
    processed_review_final = " ".join(processed_review_initial)
    if len(processed_review_final) == 0:
        review_processed_final = ''
    data.append(processed_review_final)

# prepare output label
sentiments = []
for i in range(len(sentiments_initial)):
    ans = sentiments_initial[i]
    if ans == 'negative':
        sentiments.append(0)
    else:
        sentiments.append(1)

# split train and test data
X_train, X_test, y_train_final, y_test_actual = train_test_split(reviews, sentiments, train_size=0.9)

print(y_train_final)

# extract features from text
if vectorizerValue == 1:
    vectorizer = TfidfVectorizer(min_df=10, token_pattern=r'[a-zA-Z]+')
elif vectorizerValue == 2:
    vectorizer = CountVectorizer(min_df=10, token_pattern=r'[a-zA-Z]+')
else:
    vectorizer = TfidfVectorizer(min_df=10, token_pattern=r'[a-zA-Z]+')
X_train_final = vectorizer.fit_transform(X_train)  # fit train
X_test_final = vectorizer.transform(X_test)  # transform test

# Classifier
if classifierValue == 1:
    classifier = GaussianNB()
elif classifierValue == 2:
    classifier = LinearRegression()
elif classifierValue == 3:
    classifier = LogisticRegression()
elif classifierValue == 4:
    classifier = svm.SVC(kernel='linear')
elif classifierValue == 5:
    classifier = svm.SVC(kernel='poly', degree=2)
elif classifierValue == 6:
    classifier = svm.SVC(kernel='poly', degree=3)
else:
    classifier = GaussianNB()
classifier.fit(X_train_final.toarray(), y_train_final)

# predict class
y_test_pred = classifier.predict(X_test_final.toarray())

# confusion matrix
confusion_matrix_var = confusion_matrix(y_test_actual, y_test_pred)
classification_report_var = classification_report(y_test_actual, y_test_pred)
print(confusion_matrix_var)
print(classification_report_var)
