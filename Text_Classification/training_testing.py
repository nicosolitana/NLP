from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier


def SetTrainTest(xtrain, xtest, ytrain, ytest):
    global X_train, X_test, y_train, y_test
    X_train = xtrain
    X_test = xtest
    y_train = ytrain
    y_test = ytest


def TrainTestModel(classifier):
    global y_pred
    k_fold = KFold(n_splits=5)
    cls = classifier()
    print(cross_val_score(cls, X_train, y_train,
                          cv=k_fold, scoring='accuracy', n_jobs=-1))

    model = cls.fit(X_train, y_train)
    y_pred = model.predict(X_test)


def GetModelPerformance():
    accuracy = round((y_pred == y_test).sum()/len(y_pred), 3)
    cf_matrix = confusion_matrix(y_test, y_pred)
    fscore = round(
        f1_score(y_test, y_pred, pos_label='female', average='binary'), 3)
    precision = round(precision_score(
        y_test, y_pred, pos_label='female', average='binary'), 3)
    recall = round(recall_score(
        y_test, y_pred, pos_label='female', average='binary'), 3)
    return accuracy, cf_matrix, fscore, precision, recall


def GetModelPerformance_Multilabel():
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    cf_matrix = multilabel_confusion_matrix(
        y_test, y_pred)
    fscore = round(
        f1_score(y_test, y_pred, average='micro'), 3)
    precision = round(precision_score(
        y_test, y_pred, average='micro'), 3)
    recall = round(recall_score(
        y_test, y_pred, average='micro'), 3)
    return accuracy, cf_matrix, fscore, precision, recall
