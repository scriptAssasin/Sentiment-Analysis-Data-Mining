import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# from nrclex import NRCLex
nltk.download('vader_lexicon')
cachedStopWords = stopwords.words("english") 
from collections import Counter
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt


def remove_links(x):
    return re.sub(r"http\S+", "", x)

def remove_punctuation(x):
    return re.sub(r'[^\w\s]', '', x)

def lowercase(x):
    return x.lower()

def remove_stopwords(x):
    return ' '.join([word for word in x.split() if word not in cachedStopWords])


# open a file, where you stored the pickled data
file = open('eclass_all_with_sentiment_v2.pkl', 'rb')

# # dump information to that file
data = pickle.load(file)
# data.to_csv('testing1.csv', encoding='utf-8')
data['text'] = data['text'].apply(remove_links)
data['text'] = data['text'].apply(lowercase)
data['text'] = data['text'].apply(remove_punctuation)
data['text'] = data['text'].apply(remove_stopwords)
# data = data['text'].map(remove_stopwords)

# print(data['text'].head())
# # close the file
file.close()
# print(data)

#katanomi synaisthimatwn


#evresi twn pio syxna xrisomopoioymenwn leksewn
# word_count = Counter(" ".join(data['text']).split()).most_common(10)
# word_frequency = pd.DataFrame(word_count, columns = ['Word', 'Frequency'])
# print(word_frequency)

# word_frequency.plot(x='Word',y='Frequency',kind='bar')
# plt.title("Word frequency")
# plt.show()

####################################################

# astraz = data[data['text'].str.contains("astrazeneca")]

# other = data[data['text'].str.contains("moderna") & (data['text'].str.contains("pfizer") | data['text'].str.contains("biontech"))]

#print (astraz['text'].iloc[2])

# mean1 = astraz['sentiment'].value_counts().idxmax()
# mean2 = other['sentiment'].value_counts().idxmax()
# print(mean1)
#print (other['text'].iloc[2])



####################################################

# telis =  data.groupby(by=lambda x: "%d/%d" % (x.week(), x.year())).date.value_counts()
# print(telis)


# data['date']= pd.to_datetime(data['date'])

# tels = data['date'].groupby([data.date.dt.mobth]).agg('count')
# # ax = tels.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
# # plt.tight_layout()
# print(tels)
data['date']= pd.to_datetime(data['date'])

tels = data['date'].groupby([data.date.dt.year, data.date.dt.month]).agg('count')
ax = tels.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()
plt.show()


#########################################################

# # close the file
# file.close()

# print('Showing the pickled data:')

# cnt = 0
# for item in data:
#     print('The data ', cnt, ' is : ', item)
#     cnt += 1
