import nltk
import numpy as np
import re
import string


def download_necessary_functions():
    global stopwords, wn, wcorp
    stopwords = nltk.corpus.stopwords.words('english')
    wn = nltk.WordNetLemmatizer()
    wcorp = set(nltk.corpus.words.words())


def clean_text(text):
    text = "".join([word.lower()
                   for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    en_tokens = [word.lower() for word in tokens if word in wcorp]
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text
