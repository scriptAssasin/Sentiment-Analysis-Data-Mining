import numpy as np
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import datasets


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
# print(word_frequency_all)

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
astraz.to_csv('testing1.csv', encoding='utf-8')

other = data[data['text'].str.contains("moderna") & (data['text'].str.contains("pfizer") | data['text'].str.contains("biontech"))]
other.to_csv('testing2.csv', encoding='utf-8')

mean1 = astraz['sentiment'].value_counts().idxmax()
mean2 = other['sentiment'].value_counts().idxmax()

#mean1 is equal to mean2

# print(mean1,mean2) 

###################################################

#Subplots για το πληθος των tweets ανα μήνα
data['date']= pd.to_datetime(data['date'])

tels = data['date'].groupby([data.date.dt.year, data.date.dt.month]).agg('count')
ax = tels.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
plt.tight_layout()
# plt.show()



########################---------PART 3-----------#######################





#############################################


use_data = data.iloc[:22000,:]
rest = data.iloc[22000:,:]
#print(use_data)
#print(rest)

data1 = use_data.values
#print(data)
X, y = data1[:, 10], data1[:, -1]
#print(X)
#print(y)
#print(X.shape,y.shape)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#######################################################

vectorizer = CountVectorizer()
X1 = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.20, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# X2 = vectorizer.fit_transform(X_test)
print(X1.shape)

output = open('bagwords.pkl', 'wb')
pickle.dump(X1, output)
output.close()

file = open('bagwords.pkl', 'rb')

# # dump information to that file
# test = pickle.load(file)
#print(X1.toarray())

#############################################


# from sklearn.feature_extraction.text import TfidfVectorizer
# v = TfidfVectorizer()
# x = v.fit_transform(X_train)
# xt = v.fit_transform(X_test)


# output = open('tfidf.pkl', 'wb')
# pickle.dump(x, output)
# output.close()

# file = open('tfidf.pkl', 'rb')

# # # dump information to that file
# test = pickle.load(file)


#print(test.toarray())


#########################################################################

from gensim.models import Word2Vec


#print("test")
# df_X_train = pd.DataFrame(X_train, columns=['text'])
# tokenized_tweet = df_X_train['text'].apply(lambda x: x.split())  ####?
# #print(tokenized_tweet.iloc[0])
# #print("test2")

# model_w2v = Word2Vec(
#                 tokenized_tweet,
#                 vector_size=200, # desired no. of features/independent variables
#                 window=5, # context window size
#                 min_count=1,
#                 sg = 1, # 1 for skip-gram model
#                 hs = 0,
#                 negative = 10, # for negative sampling
#                 workers= 3, # no.of cores
#                 seed = 34)

# model_w2v.train(tokenized_tweet, total_examples= len(df_X_train), epochs=20)
# # #print("end")
# model_w2v.save('telis')
# retrieved_model = Word2Vec.load('telis')
# tweet_list = []

# for tweet in X_train:
#     word_tokens = tweet.split()
#     #print(word_tokens)
#     sum = retrieved_model.wv[word_tokens[0]]
#     for count,token in enumerate(word_tokens,start=1):
#         sum = np.add(sum,retrieved_model.wv[token])
#     avg = np.true_divide(sum,len(word_tokens))    
#     tweet_list.append(avg)
#     #print(avg,type(avg),avg.shape)
#     #break    


#print(tweet_list)
# output = open('w2v.pkl', 'wb')
# pickle.dump(tweet_list, output)
# output.close()


iris = datasets.load_iris()
digits = datasets.load_digits()

from sklearn import svm
print("1")
clf = svm.SVC()
print("2")
clf.fit(X_train, y_train)
print("3")
svm.SVC()
print("4")

print(clf.predict(X_test))
print("5")

# w2v_df = pd.DataFrame(tweet_list,columns=['average'])

#print(model_w2v.wv['proceeds'])

# print(retrieved_model.wv['astrazeneca'])
# print(type(retrieved_model.wv['astrazeneca']))
# vector1 = retrieved_model.wv['astrazeneca']
# vector2 = retrieved_model.wv['vaccine']
# sum = np.add(vector1,vector2)
# avg = np.true_divide(sum,2)
# print(avg)
# print(avg.shape)
# print(sum)
# print(type(sum))
# print(sum.shape,vector1.shape,vector2.shape)
#avg = sum 

#print("type = ",type(model_w2v))
#print(retrieved_model.wv.most_similar(positive="vaccine"))
#########################################################################

file.close()
# print(x.toarray())