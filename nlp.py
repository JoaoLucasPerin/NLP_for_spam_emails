# Email Spam Classification Using Natural Language Processing (NLP):

#-----------------------
# import libraries
#-----------------------

#Load-data Libraries
import pandas as pd

#Text Processing libraries
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

#Model libraries
#from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split #, learning_curve
from sklearn.feature_extraction.text import CountVectorizer # , TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Model selection / evaluation
from sklearn.model_selection import KFold, cross_val_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import plot_roc_curve
#from sklearn.metrics import plot_confusion_matrix
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

#Save the model
#import joblib
#from joblib import dump, load
#import pickle

#Evaluate the model

#Data Exploration
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np


#----------------------------------
# 1) ETL process
#----------------------------------

#----------------------------------
## Extract
#----------------------------------

# 1. Loading dataset
data = pd.read_csv('Email_spam.csv')

# 2. Exploring dataset
print(data.head(5))

#Spam Example
print('Spam example:')
print(data.loc[ data['spam'] == 1,'text'].iloc[0])

#Non-Spam example
print('Non-spam example:')
print(data.loc[ data['spam'] == 0,'text'].iloc[0])

#Shape of the dataset
print("This dataset has",data.shape[0],"rows and",data.shape[1],"columns")

# Freq analysis of spam
print(data['spam'].value_counts())
print(data['spam'].value_counts(normalize=True))

# Bar graph of spam frequency
#data['spam'].value_counts().head(32).plot(kind='bar', figsize=(7,10))

#----------------------------------
## Transform
#----------------------------------

# 1. Missing Values

#Check if we have missing values
print("Checking if we have missings...")
print(data.isnull().sum()) # 0

# 2. Duplicates

#Check if we have duplicates values
print(str(data.duplicated().sum())+" rows will be removed...") # 33

# drop duplicates
data.drop_duplicates(inplace=True)

# 3. Remove the 'subject' word from the beginning of each email
#print(data)

#print(data.columns)

data['text'] = data['text'].str.replace('Subject: ','')

# change the name of columns
#data.rename(columns = {'spam': 'Spam', 'text':'Email'}, inplace=True)

#check if only the word subject has removed from text
print("Subject: taked off:")
print(data.head())

#--------------------------------
# 2) NLP: Text Analysis / Processing
#--------------------------------

# Optative exploratories:
#Function that tokenizes each and every email into words and returns it's length
#def count_words(text):
#    words = word_tokenize(text)
#    return len(words)

#Applying the function to df['text'] and storing the count in another column
#data['count'] = data['text'].apply(count_words)
#print(data.head())

#print("Take a look of how spams have more words...")
#print(data.groupby('spam')['count'].mean())

#Function to Process the text data, remove punctuation and stop words 
def process_text(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    
    return ' '.join([word for word in no_punc.split() if word.lower() not in stopwords.words('english')])

# apllying it
data['text']=data['text'].apply(process_text)

print("Text processed without punctuation and stop words:")
print(data['text'])

# After cleaning the text. We will now carry out the process of Stemming to reduce infected words to their root
stemmer = PorterStemmer()

# Function to Stemming
def stemming (text):
    return ''.join([stemmer.stem(word) for word in text])

# apllying it
data['text']=data['text'].apply(stemming)

print("Stemmed:")
print(data.head())

# Now we will use Count Vectorizer to convert string data into Bag of Words ie Known Vocabulary
vectorizer = CountVectorizer()

message_bow = vectorizer.fit_transform(data['text'])

#print('Cleaned with vectorizer:')
#print(message_bow)

#-----------------------
# 3) Splitting the Data - train vs test
#-----------------------

X_train,X_test,y_train,y_test = train_test_split(message_bow,data['spam'],test_size=0.20)

#------------------------
# 4) Creating the Model and it's Evaluation
#------------------------

nb = MultinomialNB()

nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

#print(classification_report(y_test,y_pred))
#plot_roc_curve(nb,X_test,y_test)
#plot_confusion_matrix(nb,X_test,y_test)

# for reproductibility:
our_seed = 1234
k = 5

kfold = KFold(n_splits=k,shuffle=True,random_state=our_seed)
print("Accuracy using Cross Validation is: ",np.mean(cross_val_score(nb,message_bow,data['spam'],cv=kfold,scoring="accuracy"))*100," %")

# Model Evaluation:
# Accuracy using Cross Validation is: 98.99912203687444  %


####################################################################################################################
# Referencies: 
# https://medium.datadriveninvestor.com/email-classification-using-natural-language-processing-nlp-ee3573bc79f7
# https://www.kaggle.com/code/harshsinha1234/email-spam-classification-nlp/notebook
####################################################################################################################