# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import CountVectorizer  # 词频矩阵

from sklearn.multiclass import OneVsRestClassifier  # 结合SVM的多分类组合辅助器
import sklearn.svm as svm  # SVM辅助器


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


if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    print("Start running time: %s" % start)
  # 你运行的程序，也就是你的代码块，或者函数


    stopwords_filePath = "/code/00DataSet\\停用词01.txt"
    stopWordList = getStopWord(stopwords_filePath)


    # 训练集的特征、标签列表
    fileTrainPath1 = "B:\\code_master\\code\\00DataSet\\dataTrainSet_2To200"
    dataTrain_list_2To200, labelTrain_list_2To200 = getDataLabelList(fileTrainPath1)
    # 测试集的特征、标签列表
    fileTestPath1 = "B:\\code_master\\code\\00DataSet\\dataTestSet_2To200"
    dataTest_list_2To200, labelTest_list_2To200 = getDataLabelList(fileTestPath1)

    # 训练集的特征、标签列表
    fileTrainPath2 = "B:\\code_master\\code\\00DataSet\\dataTrainSet_2To50"
    dataTrain_list_2To50, labelTrain_list_2To50 = getDataLabelList(fileTrainPath2)
    # 测试集的特征、标签列表
    fileTestPath2 = "B:\\code_master\\code\\00DataSet\\dataTestSet_2To50"
    dataTest_list_2To50, labelTest_list_2To50 = getDataLabelList(fileTestPath2)


    vectorizer = CountVectorizer(stop_words=stopWordList, max_features=5000)
    # 其他类别专用分类，该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    '''第一个模型'''
    # 对训练集向量化
    cipinTrain = vectorizer.fit_transform(dataTrain_list_2To200)
    tfidfTrain_2To200 = transformer.fit_transform(cipinTrain)  # if-idf中的输入为已经处理过的词频矩阵
    print("(tfidfTrain.toarray()).shape : ")
    print((tfidfTrain_2To200.toarray()).shape)
    # 对测试集向量化
    cipinTest = vectorizer.transform(dataTest_list_2To200)
    tfidfTest_2To200 = transformer.transform(cipinTest)
    print("(tfidfTest.toarray()).shape : ")
    print((tfidfTest_2To200.toarray()).shape)
    '''第二个模型'''
    # 对训练集向量化
    cipinTrain_2To50 = vectorizer.fit_transform(dataTrain_list_2To50)
    tfidfTrain_2To50 = transformer.fit_transform(cipinTrain_2To50)  # if-idf中的输入为已经处理过的词频矩阵
    print("(tfidfTrain.toarray()).shape : ")
    print((tfidfTrain_2To50.toarray()).shape)
    # 对测试集向量化
    cipinTest_2To50 = vectorizer.transform(dataTest_list_2To50)
    tfidfTest_2To50 = transformer.transform(cipinTest_2To50)
    print("(tfidfTest.toarray()).shape : ")
    print((tfidfTest_2To50.toarray()).shape)

    '''实例化和训练模型'''
    # 第一个模型模型实例化
    SVM_model_2To200 = OneVsRestClassifier(svm.SVC(kernel='linear'))  # 实例化模型
    SVM_model_2To200 = SVM_model_2To200.fit(tfidfTrain_2To200, labelTrain_list_2To200)
    # 第二个模型模型实例化
    SVM_model_2To50 = OneVsRestClassifier(svm.SVC(kernel='linear'))  # 实例化模型
    SVM_model_2To50 = SVM_model_2To50.fit(tfidfTrain_2To50, labelTrain_list_2To50)
    '''保存模型'''
    import pickle
    with open('Experiment2.model', 'wb') as file:
        save = {
            'vectorizer_E2': vectorizer,
            'transformer_E2': transformer,
            'labelTrain_list_2To200': labelTrain_list_2To200,
            'labelTest_list_2To200': labelTest_list_2To200,
            'tfidfTrain_2To200': tfidfTrain_2To200,
            'tfidfTest_2To200': tfidfTest_2To200,
            'SVM_model_2To200': SVM_model_2To200,
            'labelTrain_list_2To50': labelTrain_list_2To50,
            'labelTest_list_2To50': labelTest_list_2To50,
            'tfidfTrain_2To50': tfidfTrain_2To50,
            'tfidfTest_2To50': tfidfTest_2To50,
            'SVM_model_2To50': SVM_model_2To50
        }
        pickle.dump(save, file)






