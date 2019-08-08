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
is_nltk = True
max_vocabulary = 300

data, label, len_train, len_test, le = data_process(train_fm, test_fm, task, is_nltk=is_nltk)
data = feature_extract(data, max_vocabulary).todense()
label = np.array(label).reshape(-1, 1)

x_train, y_train, x_test, y_test = data[:len_train], label[:len_train], data[len_train:], label[len_train:]

start = time.time()
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(x_train, y_train)
train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)
end = time.time()

test_data = pd.read_csv(test_fm, header=None)
id = np.array(list(test_data.iloc[:, 0])).reshape(-1, 1)
pred = np.array(le.inverse_transform(test_predictions)).reshape(-1, 1)
output = pd.DataFrame(np.concatenate([id, pred], axis=1))
output.to_csv("output.txt", header=None, index=False, sep=' ')

# evaluate(train_predictions, test_predictions, y_train, y_test)
# print("running time", end - start)
