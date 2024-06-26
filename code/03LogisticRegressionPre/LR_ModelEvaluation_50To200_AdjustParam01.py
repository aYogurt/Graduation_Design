# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import CountVectorizer  # 词频矩阵
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier  # 结合SVM的多分类组合辅助器
import sklearn.svm as svm  # SVM辅助器
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn import metrics
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
                                 # , min_df=0.02
                                 # , max_df=0.5
                                 , max_features= 5000
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

    solver_list = ['newton-cg', 'lbfgs', 'sag', 'saga']
    for i in range(len(solver_list)):
        print("调参solver: ", solver_list[i])
        #用训练集训练模型
        # model = OneVsRestClassifier(svm.SVC(kernel='linear')) #实例化模型
        LR_model = LogisticRegression(
                                       multi_class="multinomial"
                                      # multi_class=multi_class
                                      , solver= solver_list[i]
                                      # , random_state=2
                                      )
        LR_model = LR_model.fit(tfidfTrain, labelTrain_list)

        ytrain_pred = LR_model.predict(tfidfTrain)
        print("对训练集进行模型评估, Start running time: %s" % datetime.datetime.now())
        print("accuracy_score :", metrics.accuracy_score(labelTrain_list, ytrain_pred))
        print("")


        ytest_pred = LR_model.predict(tfidfTest)
        print("对测试集集进行模型评估, Start running time: %s" % datetime.datetime.now())
        print("accuracy_score :" , metrics.accuracy_score(labelTest_list, ytest_pred))
        print("")



if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    print("Start running time: %s" % start)
  # 你运行的程序，也就是你的代码块，或者函数


    stopwords_filePath = "/code/00DataSet\\停用词01.txt"
    stopWordList = getStopWord(stopwords_filePath)


    # 训练集的特征、标签列表
    fileTrainPath = "B:\\code_master\\code\\00DataSet\\dataTrainSet_50To200"
    dataTrain_list, labelTrain_list = getDataLabelList(fileTrainPath)
    # 测试集的特征、标签列表
    fileTestPath = "B:\\code_master\\code\\00DataSet\\dataTestSet_50To200"
    dataTest_list, labelTest_list = getDataLabelList(fileTestPath)

    start02 = datetime.datetime.now()
    print("getPreResult() Start running time: %s" % start02)
    getPreResult(dataTrain_list, labelTrain_list,dataTest_list,labelTest_list, stopWordList)
    end02 = datetime.datetime.now()
    print("getPreResult() End running time: %s" % end02)
    print('Running time: %s Seconds' % (end02 - start02))
    print("")

    end = datetime.datetime.now()
    print("End running time: %s" % end)
    print('Running time: %s Seconds' % (end - start))






