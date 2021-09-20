# Source: https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from data_vectorizer import *


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, wordLst):
    coordinates = []
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        for i, word in enumerate(words):
            if(word in wordLst):
                x = embeddings[:, 0]
                y = embeddings[:, 1]
                data = []
                data.append(x[i])
                data.append(y[i])
                data.append(word)
                coordinates.append(data)
    return coordinates


# Get the analogous words based on the input three words
def visualize_analogy(model,firstWord, secondWord, thirdWord, n):
    # Gets analogy
    tuplLst = get_analogy(model, firstWord, secondWord, thirdWord, n)

    # Get vectors of the analogous words
    wordLst = [seq[0] for seq in tuplLst]
    lst = []
    word_clusters = []
    for w in wordLst:
        vc = model.wv[w]
        data = []
        data.append(w)
        word_clusters.append(w)
        arr = np.concatenate((data, vc))
        lst.append(arr)

    df = pd.DataFrame(lst)
    N = 300
    Y = df.iloc[:, -N:]
    X = df[0]

    # Reduce vector size to 2 dimensional space
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    Z = tsne.fit_transform(Y)
    pdf = pd.DataFrame(Z)
    wdf = pd.DataFrame(X)

    # Return the analogous words and coordinates
    tdf = pd.concat([wdf, pdf], axis=1)
    tdf.columns = ['words', 'x', 'y']
    return tdf


# Get the similar words of the input
def visualize_similarity(model, word, n):
    # Gets analogy
    tuplLst = get_nearest_neighbor(model, word, n)

    # Get vectors of the analogous words
    wordLst = [seq[0] for seq in tuplLst]
    keys = []
    keys.append(word)
    title = ''

    # Reduce vector size to 2 dimensional space
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.wv.most_similar(word, topn=30):
            words.append(similar_word)
            embeddings.append(model.wv[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2,
                            init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(
        embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    coordinates = tsne_plot_similar_words(title, keys, embeddings_en_2d,
                                          word_clusters, 0.7, wordLst)
    
    # Return the similar words and coordinates
    cdf = pd.DataFrame(coordinates)
    cdf.columns = ['x','y','word']
    return cdf
