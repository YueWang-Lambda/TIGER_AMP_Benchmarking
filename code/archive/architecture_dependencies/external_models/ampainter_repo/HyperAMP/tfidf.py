import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

data = pd.read_csv('../data/all.txt', sep='\t', header=None)
name = data[0].tolist()
seq = data[1].tolist()

def get_k_idf(k):
    vectorizer = CountVectorizer(ngram_range=(k,k), analyzer='char')
    X = vectorizer.fit_transform(seq)
    word = vectorizer.get_feature_names_out()

    tf = TfidfTransformer(use_idf=False).fit_transform(X).toarray()
    idf = TfidfTransformer(use_idf=True, smooth_idf=False).fit(X).idf_
    
    tfidf = tf * idf + 1
    df = pd.DataFrame(tfidf, columns=word, index=name)
    df.to_pickle('../data/tf-idf/k'+ str(k) + '_tfidf.pkl')

    k_idf = dict(zip(word, idf))
    return k_idf

ls = []
for k in [2,3,4]:
    k_idf = get_k_idf(k)
    ls.append({k: k_idf})

with open('../data/tf-idf/k_idf.pkl', 'wb') as f:
    pickle.dump(ls, f)
