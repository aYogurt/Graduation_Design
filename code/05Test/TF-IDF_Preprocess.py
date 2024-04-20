# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
texts=["昨天 在 写 论文","今天 还 在 写 论文","明天 还 要 写 论文"]
tfidf_tf = TfidfTransformer().fit_transform(CountVectorizer().fit_transform(texts))
print("CountVectorizer() + TfidfTransformer() : ")
print (tfidf_tf.toarray())
print("")

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_tv = TfidfVectorizer().fit_transform(texts)
print("TfidfVectorizer() : ")
print(tfidf_tv.toarray())




