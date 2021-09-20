import pandas as pd
import numpy as np
from data_cleaner import *
import fasttext
from gensim.models import FastText


# Loads and cleans disaster-related tweets from the dataset
def LoadData():
    data = pd.read_csv('Dataset\\FastTweet.csv')
    data['Tokens'] = data['Text'].apply(lambda x: clean_text(x))
    return data


# Loads and cleans disaster-related tweets from the dataset
# and flattens the tokens into a single array as input for 
# native facebook module for FastText model training
def LoadFlattenedData():
    data = pd.read_csv('Dataset\\FastTweet.csv')
    data['Tokens'] = data['Text'].apply(lambda x: clean_text(x))
    dataArr = data["Tokens"].to_numpy()
    dataArr = np.concatenate(dataArr).ravel().tolist()
    return dataArr


# Training supervised FastText model using Gensim
def FastTweetTrainerGensim(dimension):
    download_necessary_functions()

    data = LoadData()
    model = FastText(vector_size=dimension, window=3, min_count=1, sg=1)  # instantiate
    model.build_vocab(corpus_iterable=data['Tokens'])
    model.train(corpus_iterable=data['Tokens'], total_examples=len(data['Tokens']), epochs=10)  
    model.save('pretrained_gensim\\Habagat_FastTweetGS{}.model'.format(dimension))

    print('pretrained_gensim\\Habagat_FastTweetGS{}.model has been created.'.format(dimension))


# Training supervised FastText model using Gensim
def FastTweetTrainer(dimension):
    download_necessary_functions()

    dataArr = LoadFlattenedData()
    with open("in_out_data\\outfile", "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(dataArr))

    model = fasttext.train_supervised('in_out_data\\outfile', dim=dimension, epoch=10, loss='softmax', ws=3)
    model.save_model('pretrained_ft\\Habagat_FastTweet{}.bin'.format(dimension))
    print('pretrained_ft\\Habagat_FastTweet{}.bin model has been created.'.format(dimension))


# Training unupervised FastText model using Gensim
def FastTweetTrainer_Unsupervised(dimension):
    download_necessary_functions()

    data = LoadData()
    dataArr = data["Tokens"].to_numpy()
    dataArr = np.concatenate(dataArr).ravel().tolist()
    with open("in_out_data\\outfile", "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(dataArr))

    model = fasttext.train_unsupervised('in_out_data\\outfile', dim=dimension, epoch=10, loss='softmax', ws=3)
    model.save_model('pretrained_ft_us\\Habagat_FastTweet{}.bin'.format(dimension))
    print('pretrained_ft_us\\Habagat_FastTweet{}.bin model has been created.'.format(dimension))