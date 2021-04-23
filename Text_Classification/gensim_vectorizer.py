import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def GetVectorValues(model, df):
    words = set(model.index_to_key)
    X = np.array([np.array([model[i] for i in ls if i in words])
                  for ls in df['merge']])

    X_avg = []
    for v in X:
        if v.size:
            X_avg.append(v.mean(axis=0))
        else:
            X_avg.append(np.zeros(100, dtype=float))
    X = pd.DataFrame(X_avg)
    X = X.fillna(0)
    return X


def LoadModel():
    return KeyedVectors.load_word2vec_format(
        "Pretrained_Model\\GoogleNews-vectors-negative300.bin", binary=True)


def MergePretrainedTrainedModel(model, df):
    w2v_model = Word2Vec(df['merge'], vector_size=300, min_count=1)
    total_examples = w2v_model.corpus_count
    w2v_model.build_vocab([list(model.index_to_key)], update=True)
    w2v_model.train(df['merge'], total_examples=total_examples,
                    epochs=w2v_model.epochs)
    return w2v_model


def CreateW2VModel(df):
    return Word2Vec(df['merge'], window=5, min_count=2)
