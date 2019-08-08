import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
plt.style.use('ggplot')


col_names = ["id", "text", "topic_id", "sentiment", "is_sarcastic"]
data = []
with open("dataset.tsv", encoding="utf-8") as f:
    line = f.readline()
    while line:
        line = line.rstrip('\n')
        row = line.split('\t')
        data.append(row)
        line = f.readline()
data = pd.DataFrame(data, columns=col_names)
print(len(data))
tmp = data.groupby(['topic_id']).size().reset_index().rename(columns={0: "count"})
print(tmp)
# print(data.groupby(['sentiment']).size().reset_index())
# tmp = data.groupby(['sentiment']).size().reset_index().rename(columns={0: "count"})
# plt.pie(tmp['count'], labels=["negative", "neutral", "positive"], autopct='%.1f%%', startangle=90)
# # plt.bar(range(len(tmp['topic_id'])), tmp['count'])
# # plt.xticks(range(0, 20), range(10000, 10020))
# # ax = plt.gca()
# # for tick in ax.get_xticklabels():
# #     tick.set_rotation(75)
# plt.show()
