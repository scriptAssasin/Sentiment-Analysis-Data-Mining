import pickle
import re
import nltk
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english") 

def remove_links(x):
    return x.replace('http\S+|www.\S+', '')

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

print(data['text'].head())
# # close the file
file.close()