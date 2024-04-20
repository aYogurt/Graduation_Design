# -*- coding:utf-8 -*-


import pickle

import jieba
from sklearn import metrics
from sklearn.metrics import classification_report

with open('Experiment1.model', 'rb') as file:
    tfidf_model = pickle.load(file)
    vectorizer = tfidf_model['vectorizer_E1']
    transformer = tfidf_model['transformer_E1']
    labelTest_list = tfidf_model['labelTest_list_E1']
    tfidfTest = tfidf_model['tfidfTest_E1']
    LR_model = tfidf_model['LR_model']
    MulNB_model = tfidf_model['MulNB_model']
    SVM_model = tfidf_model['SVM_model']


# ytest_pred = LR_model.predict(tfidfTest)
# print("classification_report: ")
# print(classification_report(labelTest_list, ytest_pred))
# print("\n")
# print("accuracy_score :" , metrics.accuracy_score(labelTest_list, ytest_pred))
# print("precision_score : ", metrics.precision_score(labelTest_list, ytest_pred, average='macro'))
# print("recall_score : ", metrics.recall_score(labelTest_list, ytest_pred, average='macro'))
# print("f1_score : ", metrics.f1_score(labelTest_list, ytest_pred, average='macro'))

while 1:
    print('请输入需要预测的文本:')
    a = input()
    sentence_in = [' '.join(jieba.cut(a))]
    b = vectorizer.transform(sentence_in)
    c = transformer.transform(b)
    prd = SVM_model.predict(c)
    # print(prd)  # ['故意伤害罪']
    print('预测类别：', prd[0])
    add = input('是否还要预测?(y/n):')
    if add == 'y':
        continue
    elif add == 'n':
        break

