import emoji
import nltk
import numpy as np
import re
import preprocessor as tweet_preprocess
import string
import stopwordsiso as stopwords_iso
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def download_necessary_functions():
    global stopwords, stopwords_tl, wn, wcorp, tweet_tokenizer, ps
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords_tl = stopwords_iso.stopwords("tl")  # English stopwords
    wn = nltk.WordNetLemmatizer()
    wcorp = set(nltk.corpus.words.words())
    ps = PorterStemmer()
    tweet_preprocess.set_options(
        tweet_preprocess.OPT.URL,
        tweet_preprocess.OPT.MENTION,
        tweet_preprocess.OPT.NUMBER)
    tweet_tokenizer = TweetTokenizer()
    wcorp = set(nltk.corpus.words.words())


def HashTagSegmentation(word):
    if(word.startswith('#')):
        word = word.replace('#', '')
    word = re.sub('([A-Z][a-z]+)', r' \1',
                  re.sub('([A-Z]+)', r' \1', word)).split()
    return word


def RemoveNumber(word):
    res = any(map(str.isdigit, word))
    if res:
        return ''
    else:
        return word


def LemmatizeEnglishWords(word):
    if word in wcorp:
        # return ps.stem(wn.lemmatize(word))
        return wn.lemmatize(word)
    else:
        return word


def RemoveShortWords(word):
    if len(word) > 3:
        return word
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    if(regex.search(word) != None):
        return word
    return ''

# Reference:
# https://stackoverflow.com/questions/16453522/how-can-i-detect-laughing-words-in-a-string


def ReplaceLaugh(word):
    term = 'Haha'
    regex = re.compile(
        r'\b(h*a*ha+h[ha]*)\b')
    if(regex.search(word) != None):
        return term

    return word


def SplitByDot(word):
    return word.split(".")


def clean_text(text):
    tokens = []
    try:
        text = tweet_preprocess.clean(text)
        tokens = np.array(tweet_tokenizer.tokenize(text))
        tokens = [HashTagSegmentation(word) for word in tokens]
        tokens = np.hstack(np.array(tokens, dtype=object))
        tokens = [SplitByDot(word) for word in tokens]
        tokens = np.hstack(np.array(tokens, dtype=object))
        tokens = [ReplaceLaugh(word) for word in tokens]
        tokens = [LemmatizeEnglishWords(word) for word in tokens]
        tokens = [RemoveShortWords(word) for word in tokens]
        tokens = [word for word in tokens if word not in stopwords]
        tokens = [word for word in tokens if word not in stopwords_tl]
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = list(filter(None, tokens))
        return list(map(str.lower, tokens))
    except:
        return tokens
