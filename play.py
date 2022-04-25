def main():
        
    import gensim
    import gensim.corpora as corpora
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from gensim.models.ldamodel import LdaModel

    from pprint import pprint
    import nltk
    from nltk.corpus import stopwords
    # from nrclex import NRCLex
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    cachedStopWords = stopwords.words("english") 
    from nltk.stem import PorterStemmer
    # import spacy

    import pickle
    import re 
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    import matplotlib.pyplot as plt 
    import pandas as pd

    def remove_links(idf_train):
        return re.sub(r"http\S+", "", idf_train)

    def remove_punctuation(idf_train):
        return re.sub(r'[^\w\s]', '', idf_train)

    def lowercase(idf_train):
        return idf_train.lower()

    def remove_stopwords(idf_train):
        return ' '.join([word for word in idf_train.split() if word not in cachedStopWords])

    def tokenizing(x):
        return x.split()

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

    # open a file, where you stored the pickled data
    file = open('eclass_all_with_sentiment_v2.pkl', 'rb')

    # data = pickle.load(file)

    data['text'] = data['text'].apply(remove_links)
    data['text'] = data['text'].apply(lowercase)
    data['text'] = data['text'].apply(remove_punctuation)
    data['text'] = data['text'].apply(remove_stopwords)
    # data['text'] = data['text'].apply(tokenizing)
    data['text'] = data['text'].apply(lemmatize_text)
    ps = PorterStemmer()
    data['text'] = data['text'].apply(lambda x: [ps.stem(y) for y in x])


    file = open('lda.pkl', 'rb')
    data = pickle.load(file)
    data = data.iloc[:22000,:]
    # print(data['text'])

    tweets = data['text'].values.tolist()
    id2word = Dictionary(tweets)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in tweets]
    # Build LDA model
    lda_model = LdaModel(corpus=corpus,
                    id2word=id2word,
                    num_topics=10, 
                    random_state=0,
                    chunksize=100,
                    alpha='auto',
                    per_word_topics=True)

    # pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    print(lda_model.print_topics(num_topics=6, num_words=5))

    # pyLDAvis.enable_notebook()
    visualisation  = gensimvis.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

    coherence_model_lda = CoherenceModel(model=lda_model, texts=tweets, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


if __name__ == "__main__":
    main()