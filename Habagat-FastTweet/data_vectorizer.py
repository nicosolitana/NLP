import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_model


# For gensim created models
def LoadFastTweet(path):
    return FastText.load(path)


def GetFastTweetVectorValues(model, df):
    words = set(model.wv.index_to_key)
    X = np.array([np.array([model.wv[i] for i in ls if i in words])
                  for ls in df['Tokens']])

    X_avg = []
    for v in X:
        if v.size:
            X_avg.append(v.mean(axis=0))
        else:
            X_avg.append(np.zeros(100, dtype=float))
    X = pd.DataFrame(X_avg)
    X = X.fillna(0)
    return X


# For fastText created models
def LoadFastTweetModel(path):
    return load_facebook_model(path)


def GetFastTextVectorValues(model, df):
    words = set(model.wv.index_to_key)
    X = np.array([np.array([model.wv[i] for i in ls if i in words])
                  for ls in df['Tokens']])

    X_avg = []
    for v in X:
        if v.size:
            X_avg.append(v.mean(axis=0))
        else:
            X_avg.append(np.zeros(100, dtype=float))
    X = pd.DataFrame(X_avg)
    X = X.fillna(0)
    return X


# Gets similar words of the input based on the model
def get_nearest_neighbor(model, word, n):
    return model.wv.most_similar(word, topn=n)


# Gets analogous word based on the three input words and model
# Follows the format: man : woman :: king : queen
# where firstWord is the man, secondWord is the king and thirdWord is the woman.
# This function will output the queen and its nearest neighbors
def get_analogy(model, firstWord, secondWord, thirdWord, n):
    return model.wv.most_similar(positive=[thirdWord, secondWord], negative=[firstWord], topn=n)
