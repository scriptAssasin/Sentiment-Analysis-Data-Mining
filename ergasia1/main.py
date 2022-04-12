import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

def remove_links(x):
    try:
        return x.str.replace('http\S+|www.\S+', '', regex=True)
        # return y
    except:
        return x

def remove_punctuation(x):
    try:
        return x.str.replace(r'[^\w\s]+','', regex=True)
    except:
        return x

def remove_stopwords(x):
    try:
        text_tokens = word_tokenize(x)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        return filtered_sentence
    except:
        return x

def lowercase(x):
    try:
        return x.str.lower()
    except:
        return x

# open a file, where you stored the pickled data
file = open('eclass_all_with_sentiment_v2.pkl', 'rb')

# # dump information to that file
data = pickle.load(file)
# data.to_csv('testing1.csv', encoding='utf-8')
data = data.apply(remove_links)
data = data.apply(lowercase)
data = data.apply(remove_punctuation)
# data = data['text'].map(remove_stopwords)

print(data)
# # close the file
file.close()