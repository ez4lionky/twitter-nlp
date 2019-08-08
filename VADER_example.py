import pandas as pd
# in anaconda: pip install vaderSentiment
"""
The Compound score is a metric that calculates the sum of all the 
lexicon ratings which have been normalized between -1(most extreme negative) and +1 
(most extreme positive). In the case above, lexicon ratings for andsupercool are 2.9and respectively1.3. 
The compound score turns out to be 0.75 , denoting a very high positive sentiment.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
key_dict = {'neg': 0, 'neu': 1, 'pos': 2}
label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

test_data = pd.read_csv("test_data.csv")
test_data['text'] = test_data['text'].str.replace(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ')
test_data['text'] = test_data['text'].str.replace(r"[^#@_$%a-zA-Z0-9 ]", "")
label = list(test_data['sentiment'].map(label_dict))

predict = []
for row in test_data['text']:
    print(row)
    score = analyser.polarity_scores(row)
    print(score)
    del score['compound']
    key_name = max(score, key=score.get)
    index = key_dict[key_name]
    predict.append(index)

count = 0
for i in range(len(predict)):
    if predict[i] == label[i]:
        count += 1
print(count)
