import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from pandas.core.common import flatten
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud, STOPWORDS


def MostFrequentWords(data, n):
    dataArr = data["Tokens"].to_numpy()
    dataArr = np.concatenate(dataArr).ravel().tolist()
    entireCorpus =  data['Tokens'].tolist()
    entireCorpus = list(flatten(entireCorpus))
    entireCorpusCount = Counter(entireCorpus)
    entireCorpusCommon = entireCorpusCount.most_common(n)
    entireCorpusCommonDf = pd.DataFrame(entireCorpusCommon, columns =['Word', 'Frequency'])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5, 8))
    
    # drawing the plot
    sns.barplot(x="Frequency", y="Word", data=entireCorpusCommonDf, ax=ax)

def PCATransform(target, X, n_components):
    ss = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(ss)
    dfpc = pd.DataFrame(principalComponents, columns=['X', 'Y'])
    dfpc['class'] = target

    plt.figure(1, figsize=(10, 10), dpi=100)
    plt.clf()
    sns.scatterplot(x='X', y='Y', hue='class',
                    palette="Paired_r", data=dfpc, linewidth=0, s=10)
    plt.show()
    return principalComponents


def tSNETransform(target, X, n_components):
    ss = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=n_components, perplexity=50, random_state=0)
    principalComponents = tsne.fit_transform(ss)
    dfpc = pd.DataFrame(principalComponents, columns=['X', 'Y'])
    dfpc['class'] = target

    plt.figure(1, figsize=(10, 10), dpi=100)
    plt.clf()
    sns.scatterplot(x='X', y='Y', hue='class',
                    palette="Paired_r", data=dfpc, linewidth=0, s=10)
    plt.show()
    return principalComponents


def KMeanTransform(target, principalComponents, clusters):
    principalComponents.shape
    principalComponents

    n_clusters = clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    kmeans.fit(principalComponents)

    cluster_labels = kmeans.labels_
    df = pd.DataFrame(principalComponents, columns=['X', 'Y'])
    df['target'] = target
    ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x='X', y='Y', hue='target',
                    palette="mako", data=df, linewidth=0, s=5)


def create_word_cloud(data):
    stopwords = set(STOPWORDS)
    comment_words = ''
    for val in data['Tokens']:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
