# -*- coding: utf-8 -*-

#create a function to add a column sentence that indicates the sentence u=id for each txt file as a preprocessing step.

import pandas as pd

def data_conversion(file_name):

  df_eng=pd.read_csv(file_name, delimiter='\t', header=None, skip_blank_lines=False)

  df_eng.columns=['tag', 'tokens']

  tempTokens=list(df_eng['tokens'])

  tempSentence = list()

  count = 1

  for i in tempTokens:

    tempSentence.append("Sentence" + str(count))

    tempTokens = list(df_eng['tokens'])

    if str(i) == 'nan':

      count = count+1

  dfSentence =  pd.DataFrame (tempSentence, columns=['Sentence'])

  result = pd.concat([df_eng, dfSentence], axis=1, join='inner')

  return result

#passing the text files to function

trivia_train=data_conversion('trivia10k13train.bio.txt')

trivia_test=data_conversion('trivia10k13test.bio.txt')

trivia_train.head()

trivia_train.shape

trivia_train.tokens.nunique()

trivia_train.isnull().sum()

trivia_train.dropna(inplace=True)

trivia_test.head()

trivia_test.shape

trivia_test.tokens.nunique()

trivia_test.isnull().sum()

trivia_test.dropna(inplace=True)

# get the distribution plot for the tags.

trivia_train[trivia_train ["tag"]!="O"]["tag"].value_counts(). plot (kind="bar", figsize=(10,5))

data=trivia_train.copy()

data1=trivia_test.copy()

data.rename(columns={"Sentence": "sentence_id", "tokens": "words","tag": "labels"}, inplace =True)

data1.rename(columns={"Sentence": "sentence_id", "tokens": "words","tag": "labels"}, inplace =True)

# Commented out IPython magic to ensure Python compatibility.
#For visualization
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

sns.set(font_scale=1)

# %matplotlib inline

# %config InlineBackend. figure_format = 'svg'

# For Modeling

from sklearn.ensemble import RandomForestClassifier

from sklearn_crfsuite import CRF, scorers, metrics

from sklearn_crfsuite.metrics import flat_classification_report

from sklearn.metrics import classification_report, make_scorer

import scipy.stats

import eli5

class Get_Sent(object):

  def __init__(self, dataset):

    self.n_sent = 1

    self.dataset = dataset

    self.empty = False

    agg_func = lambda s: [(a, b) for a,b in zip (s["words"].values.tolist(),s["labels"].values.tolist())]

    self.grouped = self.dataset.groupby("sentence_id").apply(agg_func)

    self.sentences = [x for x in self.grouped]

def get_next(self):

  try:

    s = self.grouped ["Sentence: {}".format(self.n_sent)]

    self.n_sent += 1

  except:

    return None

# calling the Get_Sent function and passing the train dataset

Sent_get= Get_Sent(data)

sentences=Sent_get.sentences

# calling the Get_Sent function and passing the test dataset

Sent_get= Get_Sent(data1)

sentences1 = Sent_get.sentences

#This is what a sentence will look like.

print (sentences1[0])

#shows the output of the Gent_set function for test data.

# feature mapping for the classifier.
import numpy as np

def create_ft(txt):

  return np.array([txt.istitle(), txt.islower(), txt.isupper(), len(txt), txt.isdigit(), txt.isalpha()])

#using the above function created to get the mapping of words for train data.

words  = [create_ft(x) for x in data ["words"].values.tolist()]

#lets take unique labels

target = data["labels"].values.tolist()

#print few words with array

print (words[:5])

#we get mapping of words as below (for first five words)

#using the above function created to get the mapping of words for test data.

words1 = [create_ft(x) for x in data1["words"].values.tolist()]

target1 = data1["labels"].values.tolist()

# Apply five-fold cross validation for the random classifier model and get the results as follows. Next, the cross_val_predict function is used. It is defined in sklearn.

#importing package

from sklearn.model_selection import cross_val_predict

# train the RF model

Ner_prediction = cross_val_predict(RandomForestClassifier(n_estimators=20), X=words, y=target, cv=10)

#import library

from sklearn.metrics import classification_report

#generate report

Accuracy_rpt = classification_report (y_pred= Ner_prediction, y_true=target,zero_division=1)

print (Accuracy_rpt)
