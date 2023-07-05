# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import re
from time import time
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

#toktok = ToktokTokenizer()
#wnl = WordNetLemmatizer()
#lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)
#stopwords_dict = {word: 1 for word in stopwords.words('english')}
stopwords_set = set(stopwords.words('english'))
CLEANHTML = re.compile('<.*?>')
CLEANPUNC = re.compile('\W+')

class TFIDF_VEC:
    def __init__(self):
        self.vocab_index = {}
        self.vocab_count = Counter()
        self.idf_values = {}
        
    def preprocess(self,document):
        # lower case
        low_words = document.lower()
        # remove punctuations
        cleaned = re.sub(CLEANHTML, '', low_words)
        cleaned = re.sub(CLEANPUNC, ' ', cleaned)
        # tokenize into words
        return cleaned
    
    def tokenize(self,doc):
        #tokenized = toktok.tokenize(doc) # word_tokenize(cleaned)
        tokenized = re.findall(r'\b\w\w+\b',doc)
        # remove stop words
        tokenized = [word for word in tokenized if word not in stopwords_set]
        # lemmatize
        #lemmatized = [lemmatize(token, get_part_of_speech(token)) for token in tokenized]
        #return lemmatized #' '.join(lemmatized) TOO SLOW
        return tokenized
    
    def fit(self, corpus):
        t = time()
        corpus = [self.preprocess(doc) for doc in corpus]
        corpus = [self.tokenize(doc) for doc in corpus]
        preprocess_time = time() - t
        print("fit preprocessing time: %0.3fs" % preprocess_time)

        for doc in corpus:
            doc_counts = Counter(doc)     
            for term in set(doc):
                # Update the global vocabulary with the current document's vocabulary
                self.vocab_index.setdefault(term,len(self.vocab_index))
            self.vocab_count.update(doc_counts)
        n_samples = len(corpus)
        for term in self.vocab_index:
            df = self.vocab_count[term]
            self.idf_values[term] = np.log((n_samples + 1.0) / (df + 1.0)) + 1.0
            
    def TF(self, corpus):
        shape = ( corpus.__len__(), len(self.vocab_count) )
        TF_MATRIX = np.zeros( shape, dtype=np.int32 )
        for doc_index, doc in enumerate(corpus):
            for term in doc:
                if term in self.vocab_index:
                    TF_MATRIX[doc_index][self.vocab_index[term]] += 1
        return TF_MATRIX
    
    def IDF(self, term, corpus):
        if term not in self.vocab_count:
            return 0.0
        return self.idf_values[term]
    
    def transform(self,corpus): # transform
        t = time()
        corpus = [self.preprocess(doc) for doc in corpus]
        corpus = [self.tokenize(doc) for doc in corpus]
        preprocess_time = time() - t
        print("transform preprocessing time: %0.3fs" % preprocess_time)
        t = time()
        TF_Matrix = csr_matrix(self.TF(corpus))
        TF_time = time() - t
        print("TF Matrix time: %0.3fs" % TF_time)
        t = time()
        IDF_Matrix = csr_matrix(np.array([self.IDF(term, corpus) for term in self.vocab_index]))
        IDF_time = time() - t
        print("IDF Matrix time: %0.3fs" % IDF_time)
        TFIDF = TF_Matrix.multiply(IDF_Matrix)
        #TFIDF_L2 = TFIDF / np.linalg.norm(TFIDF.toarray())
        return csr_matrix(TFIDF)
    
    def fit_transform(self, corpus):
        # create a tuples
        self.fit(corpus)
        return self.transform(corpus)

dataset  = pd.read_csv('C:/Users/alla0/Downloads/IMDB-Dataset/IMDB Dataset.csv', nrows = 25000) 
y = dataset.sentiment
X = dataset.review

train_X, test_X, train_Y,test_Y = train_test_split(X,y, test_size = 0.33, random_state=42)

vectorizer = TFIDF_VEC()
vectorizer_2 = TfidfVectorizer()

t = time()
X_train_tfidf = vectorizer.fit_transform(train_X.to_numpy()) #train_X.tolist()
trainfeature_extraction_time = time() - t
print("train extraction time: %0.3fs" % trainfeature_extraction_time)

t = time()
X_test_tfidf = vectorizer.transform(test_X.to_numpy())

testfeature_extraction_time = time() - t
print("test extraction time: %0.3fs" % testfeature_extraction_time)

#print(X_test_tf)

t = time()
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, train_Y)
training_time = time() - t
print("train time: %0.3fs" % training_time)

y_pred = naive_bayes_classifier.predict(X_test_tfidf)

score1 = metrics.accuracy_score(test_Y, y_pred)
dataframe = pd.DataFrame(index=[test_X, test_Y, y_pred])
dataframe.index.names = ['reviews', 'test sentiment', 'prediction sentiment']
print("accuracy:  %0.3f" % score1)




