# -*- coding:utf-8 -*-


import pickle

import jieba
from sklearn import metrics
from sklearn.metrics import classification_report

with open('Experiment2.model', 'rb') as file:
    tfidf_model = pickle.load(file)
    vectorizer = tfidf_model['vectorizer_E2']
    transformer = tfidf_model['transformer_E2']
    labelTest_list_2To200 = tfidf_model['labelTest_list_2To200']
    tfidfTest_2To200 = tfidf_model['tfidfTest_2To200']
    labelTest_list_2To50 = tfidf_model['labelTest_list_2To50']
    tfidfTest_2To50 = tfidf_model['tfidfTest_2To50']
    SVM_model_2To200 = tfidf_model['SVM_model_2To200']
    SVM_model_2To50 = tfidf_model['SVM_model_2To50']


# ytest_pred = SVM_model_2To50.predict(tfidfTest_2To50)
# print("classification_report: ")
# print(classification_report(labelTest_list_2To50, ytest_pred))
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
    prd1 = SVM_model_2To200.predict(c)
    # print(prd)  # ['故意伤害罪']
    # print('预测类别：', prd1[0])
    if prd1[0] == '其他类':
        prd2 = SVM_model_2To50.predict(c)
        print('调用第二个模型进行二次强化预测，预测类别：', prd2[0])
    else:
        print('预测类别：', prd1[0])
    add = input('是否还要预测?(y/n):')
    if add == 'y':
        continue
    elif add == 'n':
        break

