import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
plt.style.use('ggplot')
# sentiment analysis accuracy
pre_train_accuracy = [0.97, 0.73, 0.70]
pre_test_accuracy = [0.65, 0.68, 0.69]
nonpre_train_accuracy = [0.99, 0.73, 0.69]
nonpre_test_accuracy = [0.65, 0.72, 0.69]

pre_train_recall = [0.95, 0.60, 0.44]
pre_test_recall = [0.51, 0.51, 0.41]
nonpre_train_recall = [0.98, 0.59, 0.40]
nonpre_test_recall = [0.49, 0.52, 0.37]

# DT BNB MNB
pre_train_precision = [0.98, 0.67, 0.80]
pre_test_precision = [0.51, 0.54, 0.75]
nonpre_train_precision = [0.99, 0.67, 0.81]
nonpre_test_precision = [0.51, 0.60, 0.44]

pre_train_fscore = [0.96, 0.62, 0.45]
pre_test_fscore = [0.51, 0.51, 0.41]
nonpre_train_fscore = [0.99, 0.62, 0.39]
nonpre_test_fscore = [0.49, 0.55, 0.34]


label_list = ['DT', 'BNB', 'MNB']
x = [0, 0.5, 1.0]

axs = plt.subplot(2, 2, 1)
rects1 = plt.bar(x=x, height=pre_train_accuracy, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_train_accuracy, width=0.2, label="N")
plt.ylabel("Accuracy")
plt.xticks([index + 0.1 for index in x], label_list)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")

axs = plt.subplot(2, 2, 2)
rects1 = plt.bar(x=x, height=pre_train_recall, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_train_recall, width=0.2, label="N")
plt.ylabel("Recall")
plt.xticks([index + 0.1 for index in x], label_list)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
plt.legend(loc=1, bbox_to_anchor=(1, 1.40))

axs = plt.subplot(2, 2, 3)
rects1 = plt.bar(x=x, height=pre_train_precision, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_train_precision, width=0.2, label="N")
plt.ylabel("Precision")
plt.xticks([index + 0.1 for index in x], label_list)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")

axs = plt.subplot(2, 2, 4)
rects1 = plt.bar(x=x, height=pre_train_fscore, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_train_fscore, width=0.2, label="N")
plt.ylabel("F1 score")
plt.xticks([index + 0.1 for index in x], label_list)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
plt.show()
