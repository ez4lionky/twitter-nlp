import numpy as np
import argparse
import time
from sklearn.tree import DecisionTreeClassifier
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--train_fm", type=str, default="train_data.csv")
parser.add_argument("--test_fm", type=str, default="test_data.csv")
args = parser.parse_args()
train_fm = args.train_fm
test_fm = args.test_fm

task = 'topic'
is_nltk = False
max_vocabulary = 100

data, label, len_train, len_test = data_process(train_fm, test_fm, task, is_nltk)
data = feature_extract(data, max_vocabulary).todense()
label = np.array(label).reshape(-1, 1)

x_train, y_train, x_test, y_test = data[:len_train], label[:len_train], data[len_train:], label[len_train:]

start = time.time()
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(x_train, y_train)
train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)
end = time.time()

evaluate(train_predictions, test_predictions, y_train, y_test)
print("running time", end - start)
