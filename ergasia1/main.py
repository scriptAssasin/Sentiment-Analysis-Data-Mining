import pickle
import re
import nltk
from nltk.corpus import stopwords
# from nrclex import NRCLex
nltk.download('vader_lexicon')
cachedStopWords = stopwords.words("english") 
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


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

#######################################################################

#katanomi synaisthimatwn
selected_sentiments=['NEU','POS','NEG']

sentiment_distribution = data.loc[data['sentiment'].isin(selected_sentiments),'sentiment'].value_counts()
plt.bar(sentiment_distribution.index, sentiment_distribution.values)
# plt.show()

#######################################################################

#evresi twn pio syxna xrisomopoioymenwn leksewn
word_count_all = Counter(" ".join(data['text']).split()).most_common(10)
word_frequency_all = pd.DataFrame(word_count_all, columns = ['Word', 'Frequency'])
print(word_frequency_all)

word_frequency_all.plot(x='Word',y='Frequency',kind='bar')
plt.title("Word frequency")
# plt.show()

#################################################################

# #evresi twn pio syxna xrisomopoioymenwn leksewn analoga to synaisthima
negative_subset_data = data[data['sentiment'] == 'NEG']
positive_subset_data = data[data['sentiment'] == 'POS']
neutral_subset_data = data[data['sentiment'] == 'NEU']

word_count_negative = Counter(" ".join(negative_subset_data['text']).split()).most_common(10)
word_frequency_negative = pd.DataFrame(word_count_negative, columns = ['Word', 'Frequency'])
word_frequency_negative.plot(x='Word',y='Frequency',kind='bar')
plt.title("Word frequency Negative")
# plt.show()


word_count_positive = Counter(" ".join(positive_subset_data['text']).split()).most_common(10)
word_frequency_positive = pd.DataFrame(word_count_positive, columns = ['Word', 'Frequency'])
word_frequency_positive.plot(x='Word',y='Frequency',kind='bar')
plt.title("Word frequency Positive")
# plt.show()


word_count_neutral = Counter(" ".join(neutral_subset_data['text']).split()).most_common(10)
word_frequency_neutral = pd.DataFrame(word_count_neutral, columns = ['Word', 'Frequency'])
word_frequency_neutral.plot(x='Word',y='Frequency',kind='bar')
plt.title("Word frequency Neutral")
# plt.show()

#####################################################################

#Sygrisi ws pros synaisthima astrazeneca vs moderna & pfizer

astraz = data[data['text'].str.contains("astrazeneca")]

other = data[data['text'].str.contains("moderna") & (data['text'].str.contains("pfizer") | data['text'].str.contains("biontech"))]

mean1 = astraz['sentiment'].value_counts().idxmax()
mean2 = other['sentiment'].value_counts().idxmax()

#mean1 is equal to mean2

print(mean1,mean2) 

###################################################

#Subplots για το πληθος των tweets ανα μήνα
data['date']= pd.to_datetime(data['date'])

tels = data['date'].groupby([data.date.dt.year, data.date.dt.month]).agg('count')
ax = tels.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()
# plt.show()
