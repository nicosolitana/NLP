import pandas as pd
import numpy as np
from data_cleaner import *
from data_training_testing import *
from data_vectorizer import *
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def LoadData():
    data = pd.read_csv('Dataset\\DisasterPHClassification.csv')
    data['Tokens'] = data['Text'].apply(lambda x: clean_text(x))
    return data

def TextClassification(modelpath):
    download_necessary_functions()

    data = LoadData()
    print('Data has been loaded.')

    # PRE-TRAINED HABAGAT-FASTTWEET
    model = LoadFastTweetModel(modelpath)
    print('FastText PreTrained model has been loaded.')

    # Convert tokens to Vectors and split for training and testing
    print('Vectorizing tokens and feature engineering.')
    X = GetFastTextVectorValues(model, data)

    # Splitting dataset into Training and Testing
    print('Splitting data into training and testing sets.')
    X_train, X_test, y_train, y_test = train_test_split(X, data['Disaster'], test_size=0.25)

    # LOGISTIC REGRESSION
    # Model Building, Cross Validation and Prediction
    print('Training and Testing model.')
    SetTrainTest(X_train, X_test, y_train, y_test)
    TrainTestModel(lambda: LogisticRegression(solver='lbfgs',max_iter=500))

    # Check Model Performance
    accuracy, cf_matrix, f1_score, precision, recall = GetModelPerformance()

    return accuracy, cf_matrix, f1_score, precision, recall
    # print('Accuracy: {} FScore: {} Precision: {} Recall: {}'.format(accuracy, f1_score, precision, recall))


def TextClassificationGS(modelpath):
    download_necessary_functions()

    data = LoadData()
    print('Data has been loaded.')

    # PRE-TRAINED HABAGAT-FASTTWEET
    model = LoadFastTweet(modelpath)
    print('FastText PreTrained model has been loaded.')

    # Convert tokens to Vectors and split for training and testing
    print('Vectorizing tokens and feature engineering.')
    X = GetFastTextVectorValues(model, data)

    # Splitting dataset into Training and Testing
    print('Splitting data into training and testing sets.')
    X_train, X_test, y_train, y_test = train_test_split(X, data['Disaster'], test_size=0.25)

    # LOGISTIC REGRESSION
    # Model Building, Cross Validation and Prediction
    print('Training and Testing model.')
    SetTrainTest(X_train, X_test, y_train, y_test)
    TrainTestModel(lambda: LogisticRegression(solver='lbfgs',max_iter=500))

    # Check Model Performance
    accuracy, cf_matrix, f1_score, precision, recall = GetModelPerformance()

    return accuracy, cf_matrix, f1_score, precision, recall