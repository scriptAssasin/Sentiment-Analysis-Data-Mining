import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
# from nrclex import NRCLex
nltk.download('vader_lexicon')
nltk.download('stopwords')
cachedStopWords = stopwords.words("english") 
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def remove_links(idf_train):
    return re.sub(r"http\S+", "", idf_train)

def remove_punctuation(idf_train):
    return re.sub(r'[^\w\s]', '', idf_train)

def lowercase(idf_train):
    return idf_train.lower()

def remove_stopwords(idf_train):
    return ' '.join([word for word in idf_train.split() if word not in cachedStopWords])


# open a file, where you stored the pickled data
file = open('eclass_all_with_sentiment_v2.pkl', 'rb')

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

# #katanomi synaisthimatwn
# selected_sentiments=['NEU','POS','NEG']

# sentiment_distribution = data.loc[data['sentiment'].isin(selected_sentiments),'sentiment'].value_counts()
# plt.bar(sentiment_distribution.index, sentiment_distribution.values)
# # plt.show()

# #######################################################################

# #evresi twn pio syxna xrisomopoioymenwn leksewn
# word_count_all = Counter(" ".join(data['text']).split()).most_common(10)
# word_frequency_all = pd.DataFrame(word_count_all, columns = ['Word', 'Frequency'])
# # print(word_frequency_all)

# word_frequency_all.plot(idf_train='Word',expected_output='Frequency',kind='bar')
# plt.title("Word frequency")
# # plt.show()

# #################################################################

# # #evresi twn pio syxna xrisomopoioymenwn leksewn analoga to synaisthima
# negative_subset_data = data[data['sentiment'] == 'NEG']
# positive_subset_data = data[data['sentiment'] == 'POS']
# neutral_subset_data = data[data['sentiment'] == 'NEU']

# word_count_negative = Counter(" ".join(negative_subset_data['text']).split()).most_common(10)
# word_frequency_negative = pd.DataFrame(word_count_negative, columns = ['Word', 'Frequency'])
# word_frequency_negative.plot(idf_train='Word',expected_output='Frequency',kind='bar')
# plt.title("Word frequency Negative")
# # plt.show()


# word_count_positive = Counter(" ".join(positive_subset_data['text']).split()).most_common(10)
# word_frequency_positive = pd.DataFrame(word_count_positive, columns = ['Word', 'Frequency'])
# word_frequency_positive.plot(idf_train='Word',expected_output='Frequency',kind='bar')
# plt.title("Word frequency Positive")
# # plt.show()


# word_count_neutral = Counter(" ".join(neutral_subset_data['text']).split()).most_common(10)
# word_frequency_neutral = pd.DataFrame(word_count_neutral, columns = ['Word', 'Frequency'])
# word_frequency_neutral.plot(idf_train='Word',expected_output='Frequency',kind='bar')
# plt.title("Word frequency Neutral")
# # plt.show()

# #####################################################################

# #Sygrisi ws pros synaisthima astrazeneca vs moderna & pfizer

# astraz = data[data['text'].str.contains("astrazeneca")]
# astraz.to_csv('testing1.csv', encoding='utf-8')

# other = data[data['text'].str.contains("moderna") & (data['text'].str.contains("pfizer") | data['text'].str.contains("biontech"))]
# other.to_csv('testing2.csv', encoding='utf-8')

# mean1 = astraz['sentiment'].value_counts().idxmax()
# mean2 = other['sentiment'].value_counts().idxmax()

# #mean1 is equal to mean2

# # print(mean1,mean2) 

# ###################################################

# #Subplots για το πληθος των tweets ανα μήνα
# data['date']= pd.to_datetime(data['date'])

# tels = data['date'].groupby([data.date.dt.year, data.date.dt.month]).agg('count')
# ax = tels.unstack(level=0).plot(kind='bar', subplots=True, rot=0, figsize=(9, 7), layout=(2, 3))
# plt.tight_layout()
# # plt.show()



########################---------PART 3-----------#######################





#############################################

# 10% split
use_data = data.iloc[:22000,:]
rest = data.iloc[22000:,:]

model_data = use_data.values
input, expected_output = model_data[:, 10], model_data[:, -1]

# train and test split
input_train, input_test, output_train, output_test = train_test_split(input, expected_output, test_size=0.20, random_state=1)


###################################################

# bag of words
vectorizer = CountVectorizer()
bow_train = vectorizer.fit_transform(input_train)
bow_test = vectorizer.transform(input_test)

# file save
output = open('bagwords.pkl', 'wb')
pickle.dump(bow_train, output)
output.close()

#file = open('bagwords.pkl', 'rb')
#test = pickle.load(file)

#############################################

# tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
idf_train = v.fit_transform(input_train)
idf_test = v.transform(input_test)

output = open('tfidf.pkl', 'wb')
pickle.dump(idf_train, output)
output.close()

#file = open('tfidf.pkl', 'rb')
#test = pickle.load(file)

#########################################################################

# word 2 vec

df_X_train = pd.DataFrame(input_train, columns=['text'])
tokenized_tweet_train = df_X_train['text'].apply(lambda idf_train: idf_train.split()) 

df_X_test = pd.DataFrame(input_test, columns=['text'])
tokenized_tweet_test = df_X_test['text'].apply(lambda idf_train: idf_train.split())  


model_w2v = Word2Vec(
                tokenized_tweet_train,
                vector_size=200, # desired no. of features/independent variables
                window=5, # context window size
                min_count=1,
                sg = 1, # 1 for skip-gram model
                hs = 0,
                negative = 10, # for negative sampling
                workers= 3, # no.of cores
                seed = 34)

model_w2v_test = Word2Vec(
                tokenized_tweet_test,
                vector_size=200, # desired no. of features/independent variables
                window=5, # context window size
                min_count=1,
                sg = 1, # 1 for skip-gram model
                hs = 0,
                negative = 10, # for negative sampling
                workers= 3, # no.of cores
                seed = 34)
model_w2v.train(tokenized_tweet_train, total_examples = len(df_X_train), epochs=20)
model_w2v_test.train(tokenized_tweet_test, total_examples = len(df_X_test), epochs=20)

# model_w2v.save('saved_model')
#retrieved_model = Word2Vec.load('saved_model')

tweet_list_train = []
tweet_list_test = []

# from word vector to tweet vector

for tweet in input_train:
    word_tokens = tweet.split()
    sum = model_w2v.wv[word_tokens[0]]
    for count,token in enumerate(word_tokens,start=1):
        sum = np.add(sum,model_w2v.wv[token])
    avg = np.true_divide(sum,len(word_tokens))    
    tweet_list_train.append(avg)  

for tweet in input_test:
    word_tokens = tweet.split()
    sum = model_w2v_test.wv[word_tokens[0]]
    for count,token in enumerate(word_tokens,start=1):
        sum = np.add(sum,model_w2v_test.wv[token])
    avg = np.true_divide(sum,len(word_tokens))    
    tweet_list_test.append(avg)


# output = open('w2v.pkl', 'wb')
# pickle.dump(tweet_list_train, output)
# output.close()


#################################################

# part 4

# classifier selection

#clf = svm.SVC()
#clf = RandomForestClassifier()
clf = KNeighborsClassifier(n_neighbors=3)

# model train

#clf.fit(bow_train, output_train)
clf.fit(idf_train, output_train)
#clf.fit(tweet_list_train, output_train)

#svm.SVC()   # needed for svm! dont delete

# prediction

#print(clf.predict(bow_test))
print(clf.predict(idf_test))
print(clf.score(idf_test,output_test))
sklearn.model_selection.KFold
scores = cross_val_score(clf,input,expected_output,cv=10)
#scores = cross_val_score(clf,input,expected_output,cv=10, scoring = 'accuracy')
#scores = cross_val_score(clf,input,expected_output,cv=10, scoring = 'precision')
#scores = cross_val_score(clf,input,expected_output,cv=10, scoring = 'recall')
#scores = cross_val_score(clf,input,expected_output,cv=10, scoring = 'f1')

#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#print(clf.predict(tweet_list_test))

#####################################################
# tests
# print(model_w2v_test.wv.most_similar(positive="vaccine"))
# print(model_w2v.wv.most_similar(positive="vaccine"))

#######################################################
