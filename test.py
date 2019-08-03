import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
plt.style.use('ggplot')
# sentiment analysis accuracy
pre_train_accuracy = [0.82, 0.47, 0.54]
pre_test_accuracy = [0.34, 0.41, 0.42]
nonpre_train_accuracy = [0.82, 0.47, 0.43]
nonpre_test_accuracy = [0.27, 0.40, 0.38]

pre_train_recall = [0.74, 0.30, 0.29]
pre_test_recall = [0.22, 0.23, 0.21]
nonpre_train_recall = [0.75, 0.29, 0.21]
nonpre_test_recall = [0.17, 0.23, 0.18]

# DT BNB MNB
pre_train_precision = [0.87, 0.43, 0.40]
pre_test_precision = [0.22, 0.22, 0.30]
nonpre_train_precision = [0.87, 0.43, 0.29]
nonpre_test_precision = [0.18, 0.22, 0.23]

pre_train_fscore = [0.78, 0.31, 0.30]
pre_test_fscore = [0.21, 0.22, 0.21]
nonpre_train_fscore = [0.79, 0.30, 0.22]
nonpre_test_fscore = [0.17, 0.22, 0.18]


label_list = ['DT', 'BNB', 'MNB']
x = [0, 0.5, 1.0]

axs = plt.subplot(2, 2, 1)
rects1 = plt.bar(x=x, height=pre_test_accuracy, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_test_accuracy, width=0.2, label="N")
plt.ylabel("Accuracy")
plt.xticks([index + 0.1 for index in x], label_list)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")

axs = plt.subplot(2, 2, 2)
rects1 = plt.bar(x=x, height=pre_test_recall, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_test_recall, width=0.2, label="N")
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
rects1 = plt.bar(x=x, height=pre_test_precision, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_test_precision, width=0.2, label="N")
plt.ylabel("Precision")
plt.xticks([index + 0.1 for index in x], label_list)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")

axs = plt.subplot(2, 2, 4)
rects1 = plt.bar(x=x, height=pre_test_fscore, width=0.2, alpha=0.8, label="Y")
rects2 = plt.bar(x=[i + 0.2 for i in x], height=nonpre_test_fscore, width=0.2, label="N")
plt.ylabel("F1 score")
plt.xticks([index + 0.1 for index in x], label_list)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+0.05, str(height), ha="center")
plt.show()
