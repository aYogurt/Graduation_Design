# -*- coding:utf-8 -*-

from sklearn.multiclass import OneVsRestClassifier  # 结合SVM的多分类组合辅助器
import sklearn.svm as svm  # SVM辅助器
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import CountVectorizer  # 词频矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_validate

from os import listdir
import os
import jieba

from time import time
import datetime


def getStopWord(inputPath):
    with open(inputPath, "r", encoding='utf-8') as file:
        content = file.read()
        file.close()
    stopWordList = content.splitlines()
    return stopWordList


# getDataLabelList() : 取数据、标签列表
def getDataLabelList(inputPath):
    dataList = []
    labelList = []
    fatherLists = listdir(inputPath) # 目录 ['xx罪.txt','xx罪.txt',...]
    for eachDir in fatherLists:  # 遍历目录中各个文件
        eachPath = inputPath +"\\"+ eachDir  # 保存目录中每个文件路径，便于遍历二级文件  G:\\dataSetTest\\xx罪.txt
        with open(eachPath, 'r', encoding="utf-8") as f:
            content = f.readlines()
            for i in range(len(content)):
                labelList.append(eachDir.replace('.txt',''))
                dataList.append(" ".join(jieba.cut(content[i].strip())))
            f.close()
    return dataList, labelList



# getPreResult() : 输入案例并返回预测结果
def getPreResult(dataTrain_list, labelTrain_list,dataTest_list,labelTest_list, stopWordList):
    vectorizer = CountVectorizer(stop_words=stopWordList
                                 # ,max_df=0.5
                                 ,max_features=5000
                                 )  # 完善min_df的取值 ，学习曲线？？？
    # 其他类别专用分类，该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    #对训练集向量化
    cipinTrain = vectorizer.fit_transform(dataTrain_list)
    tfidfTrain = transformer.fit_transform(cipinTrain)  # if-idf中的输入为已经处理过的词频矩阵
    print("(tfidfTrain.toarray()).shape : ")
    print((tfidfTrain.toarray()).shape)
    #对测试集向量化
    cipinTest = vectorizer.transform(dataTest_list)
    tfidfTest = transformer.transform(cipinTest)
    print("(tfidfTest.toarray()).shape : ")
    print((tfidfTest.toarray()).shape)

    #用训练集训练模型
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    clf = model.fit(tfidfTrain, labelTrain_list)

    print("对训练集进行模型评估：")
    ytrain_pred = clf.predict(tfidfTrain)
    print("classification_report: ")
    print(classification_report(labelTrain_list, ytrain_pred))
    print("\n")
    scoreTrian = clf.score(tfidfTrain, labelTrain_list)
    print("score:", scoreTrian)

    print("")
    print("对测试集进行模型评估：")
    ytest_pred = clf.predict(tfidfTest)
    print("classification_report: ")
    print(classification_report(labelTest_list, ytest_pred))
    print("\n")
    scoreTest = clf.score(tfidfTest, labelTest_list)
    print("score:", scoreTest)

    # while 1:
    #     print('请输入需要预测的文本:')
    #     a = input()
    #     sentence_in = [' '.join(jieba.cut(a))]
    #     b = vectorizer.transform(sentence_in)
    #     c = transformer.transform(b)
    #     prd = clf.predict(c)
    #     # print(prd)  # ['故意伤害罪']
    #     print('预测类别：', prd[0])
    #     add = input('是否还要预测?(y/n):')
    #     if add == 'y':
    #         continue
    #     elif add == 'n':
    #         break

if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    print("Start running time: %s" % start)
  # 你运行的程序，也就是你的代码块，或者函数


    stopwords_filePath = "/code/00DataSet\\停用词01.txt"
    stopWordList = getStopWord(stopwords_filePath)



    # 训练集的特征、标签列表
    fileTrainPath = "B:\\code_master\\code\\00DataSet\\dataTrainSet_under200"
    dataTrain_list, labelTrain_list = getDataLabelList(fileTrainPath)
    # 测试集的特征、标签列表
    fileTestPath = "B:\\code_master\\code\\00DataSet\\dataTestSet_under200"
    dataTest_list, labelTest_list = getDataLabelList(fileTestPath)

    start02 = datetime.datetime.now()
    print("getPreResult() Start running time: %s" % start02)
    getPreResult(dataTrain_list, labelTrain_list,dataTest_list,labelTest_list, stopWordList)
    end02 = datetime.datetime.now()
    print("getPreResult() End running time: %s" % end02)
    print('Running time: %s Seconds' % (end02 - start02))

    # end = datetime.datetime.now()
    # print("End running time: %s" % end)
    # print('Running time: %s Seconds' % (end - start))






