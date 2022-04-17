import pickle
import re
import nltk
from nltk.corpus import stopwords
# from nrclex import NRCLex
nltk.download('vader_lexicon')
cachedStopWords = stopwords.words("english") 
from collections import Counter
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


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

analyzer = SentimentIntensityAnalyzer()
data['polarity'] = data['text'].apply(lambda x: analyzer.polarity_scores(x))
print(data.head(3))

#evresi twn pio syxna xrisomopoioymenwn leksewn
# word_count = Counter(" ".join(data['text']).split()).most_common(10)
# word_frequency = pd.DataFrame(word_count, columns = ['Word', 'Frequency'])
# print(word_frequency)

# # close the file
# file.close()

# print('Showing the pickled data:')

# cnt = 0
# for item in data:
#     print('The data ', cnt, ' is : ', item)
#     cnt += 1
