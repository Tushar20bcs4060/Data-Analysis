


import numpy as np
import pandas as pd
import os
import csv

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


# In[2]:


filename = r'C:/Users/Tushar Mani/OneDrive/Desktop/Minor Project/Sentiment Data/UkraineCombinedTweetsDeduped20220227-131611.csv.gzip'
df =  pd.read_csv(filename, compression='gzip', index_col=0,encoding='utf-8', quoting=csv.QUOTE_ALL)

print(df.shape)


# In[3]:


import math

oneFifth = math.ceil(len(df) * 0.05)


# In[4]:


df = df[df["language"] == "en"].sample(oneFifth)


# In[5]:


df.reset_index(drop=True, inplace=True)


# In[6]:


print(df.shape)


# In[7]:


import re
from bs4 import BeautifulSoup
from html import unescape

def remove_urls(x):
    cleaned_string = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str(x), flags=re.MULTILINE)
    return cleaned_string


# In[8]:


def unescape_stuff(x):
    soup = BeautifulSoup(unescape(x), 'lxml')
    return soup.text


# In[9]:


def deEmojify(x):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'', x)


# In[10]:


def unify_whitespaces(x):
    cleaned_string = re.sub(' +', ' ', x)
    return cleaned_string 


# In[11]:


def remove_symbols(x):
    cleaned_string = re.sub(r"[^a-zA-Z0-9?!.,]+", ' ', x)
    return cleaned_string  


# In[12]:


df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(remove_urls)
df['text'] = df['text'].apply(unescape_stuff)
df['text'] = df['text'].apply(deEmojify)
df['text'] = df['text'].apply(remove_symbols)
df['text'] = df['text'].apply(unify_whitespaces)


# In[13]:


print(df['text'].head())


# In[15]:


from flair.models import TextClassifier
from flair.data import Sentence
sia = TextClassifier.load('en-sentiment')


# In[16]:


print(sia)


# In[17]:


def flair_prediction(x):

    sentence = Sentence(x)
    
    try:        
        sia.predict(sentence)
        score = sentence.labels[0]
        staging_score = str(score).replace("(",",").replace(")","")
        
        sentiment_score = staging_score.split(",")
        
        if "POSITIVE" in str(sentiment_score[0]):
            return sentiment_score[0].strip(), float(sentiment_score[1].strip())
        elif "NEGATIVE" in str(sentiment_score[0]):
            return sentiment_score[0].strip(), float(sentiment_score[1].strip())
        else:
            return "NEUTRAL", 0.00
    except Exception:
        print(sentence)
        pass  # or you could use 'continue'
    
    return "ERROR",0.00


# In[18]:


df['Sentiment'] = ""
df['Sentiment_Score'] = np.nan


# In[19]:


import swifter
df["Sentiment"],df["Sentiment_Score"] =  zip(*df["text"].swifter.apply(flair_prediction))


# In[20]:


df.head().T


# In[21]:


df.groupby(["Sentiment"]).agg({'Sentiment_Score': ['count','mean']})


# In[22]:


import matplotlib.pyplot as plt

df.groupby(['Sentiment']).sum().plot(kind='pie', y="tweetid", autopct='%1.1f%%', startangle=270, fontsize=17)


# In[ ]:





# In[ ]:




