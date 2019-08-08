import argparse
import numpy as np
from load_data import *
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def get_oof(clf, n_folds, X_train, y_train, X_test):
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    classnum = len(np.unique(y_train))
    skf = StratifiedKFold(n_splits=n_folds, random_state=1)
    oof_train = np.zeros((ntrain, classnum))
    oof_test = np.zeros((ntest, classnum))

    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        skf_X_train = X_train[train_index]
        skf_y_train = y_train[train_index]

        skf_X_test = X_train[test_index]
        clf.fit(skf_X_train, skf_y_train)
        oof_train[test_index] = clf.predict_proba(skf_X_test)

        oof_test += clf.predict_proba(X_test)
    oof_test = oof_test/float(n_folds)
    return oof_train, oof_test


parser = argparse.ArgumentParser()
parser.add_argument("--train_fm", type=str, default="train_data.csv")
parser.add_argument("--test_fm", type=str, default="test_data.csv")
args = parser.parse_args()
train_fm = args.train_fm
test_fm = args.test_fm

task = 'topic'
is_nltk = True
max_vocabulary = 300

data, label, len_train, len_test, le = data_process(train_fm, test_fm, task, is_nltk=is_nltk)
data = feature_extract(data, max_vocabulary).todense()
label = np.array(label)

x_train, y_train, x_test, y_test = data[:len_train], label[:len_train], data[len_train:], label[len_train:]
ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

clfs = [LogisticRegression(),
        RandomForestClassifier(n_estimators=200, max_depth=6),
        GradientBoostingClassifier(learning_rate=0.1, max_depth=6, n_estimators=200)]

dataset_blend_train = np.zeros((x_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((x_test.shape[0], len(clfs)))
newfeature_list = []
newtestdata_list = []
for clf in clfs:
    oof_train_, oof_test_ = get_oof(clf, n_folds=10, X_train=x_train, y_train=y_train, X_test=x_test)
    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)

newfeature = reduce(lambda x, y: np.concatenate((x, y), axis=1), newfeature_list)
newtestdata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newtestdata_list)
model = LogisticRegression()
model.fit(newfeature, y_train)
train_predictions = model.predict(newfeature)
test_predictions = model.predict(newtestdata)

test_data = pd.read_csv(test_fm, header=None)
id = np.array(list(test_data.iloc[:, 0])).reshape(-1, 1)
pred = np.array(le.inverse_transform(test_predictions)).reshape(-1, 1)
output = pd.DataFrame(np.concatenate([id, pred], axis=1))
output.to_csv("output.txt", header=None, index=False, sep=' ')

# evaluate(train_predictions, test_predictions, y_train, y_test)
