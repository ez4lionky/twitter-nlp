import pandas as pd
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def rm_stopwords(x):
    data = []
    stop_words = set(stopwords.words('english'))
    for i in x:
        if i not in stop_words:
            data.append(i)
    return data


def data_process(train_fm, test_fm, task, drop_sarcastic=False, drop_neutral=False, is_nltk=True):
    col_names = ['id', 'text', 'topic_id', 'sentiment', 'is_sarcastic']
    train_data = pd.read_csv(train_fm, header=None)
    test_data = pd.read_csv(test_fm, header=None)
    if not drop_neutral:
        len_train = len(train_data)
        len_test = len(test_data)
    df = pd.concat([train_data, test_data])
    df.columns = col_names
    if drop_sarcastic:
        df = df[~df['is_sarcastic'].isin([True])]
    if drop_neutral:
        df = df[~df['sentiment'].isin(["neutral"])]
        len_train = int(0.8 * len(df))
        len_test = len(df) - len_train

    df['text'] = df['text'].str.replace(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ')
    df['text'] = df['text'].str.replace(r"[^#@_$%a-zA-Z0-9 ]", "")
    df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

    tokenized_data = df['text'].apply(lambda x: x.split())
    # stem and stop words removal
    if is_nltk:
        tokenized_data = tokenized_data.apply(rm_stopwords)
        stem = PorterStemmer()
        tokenized_data = tokenized_data.apply(lambda x: [stem.stem(i) for i in x])
    data = tokenized_data.apply(lambda x: ' '.join(x))
    if task == 'sentiment':
        if drop_neutral:
            label_dict = {'negative': 0, 'positive': 1}
        else:
            label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
        label = list(df['sentiment'].map(label_dict))
        return data, label, len_train, len_test
    elif task == 'topic':
        le = LabelEncoder()
        le.fit(df['topic_id'])
        label = le.transform(df['topic_id'])
        return data, label, len_train, len_test, le


def feature_extract(x, max_vocabulary):
    tfidf_vec = TfidfVectorizer(max_features=max_vocabulary)
    tfidf = tfidf_vec.fit_transform(x)
    return tfidf


def evaluate(train_predictions, test_predictions, y_train, y_test):
    train_accuracy = metrics.accuracy_score(y_train, train_predictions)
    train_recall = metrics.recall_score(y_train, train_predictions, average="macro")
    train_precision = metrics.precision_score(y_train, train_predictions, average="macro")
    train_precision_micro = metrics.precision_score(y_train, train_predictions, average="micro")
    train_F1 = metrics.f1_score(y_train, train_predictions, average="macro")
    # train_class = metrics.precision_score(y_train, train_predictions, average=None)

    test_accuracy = metrics.accuracy_score(y_test, test_predictions)
    test_recall = metrics.recall_score(y_test, test_predictions, average="macro")
    test_precision = metrics.precision_score(y_test, test_predictions, average="macro")
    test_precision_micro = metrics.precision_score(y_test, test_predictions, average="micro")
    test_F1 = metrics.f1_score(y_test, test_predictions, average="macro")
    # test_class = metrics.precision_score(y_test, test_predictions, average=None)

    print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)
    print("train recall", train_recall, "test recall", test_recall)
    print("train precision", train_precision, "test precision", test_precision)
    print("train precision_micro", train_precision_micro, "test precision_micro", test_precision_micro)
    print("train F1", train_F1, "test F1", test_F1)
    # print("train_class", train_class, "test_class", test_class)
