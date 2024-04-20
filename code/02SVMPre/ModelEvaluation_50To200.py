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

def getCVResult(data_list, label_list, stopWordList):
    vectorizer = CountVectorizer(stop_words=stopWordList, min_df=0.02)
    transformer = TfidfTransformer()

    #文本向量化，tf-idf处理
    cipin = vectorizer.fit_transform(data_list)
    tfidf = transformer.fit_transform(cipin)

    #实例化模型
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))

    scores = cross_val_score(model, tfidf, label_list, cv=5)
    print('cv=5的分数为: ', scores)
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(model, tfidf, label_list, scoring=scoring, cv=5, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)


# getPreResult() : 输入案例并返回预测结果
def getPreResult(dataTrain_list, labelTrain_list,dataTest_list,labelTest_list, stopWordList):
    vectorizer = CountVectorizer(stop_words=stopWordList, min_df=0.02)  # 完善min_df的取值 ，学习曲线？？？
    # 其他类别专用分类，该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    #对训练集向量化
    cipinTrain = vectorizer.fit_transform(dataTrain_list)
    tfidfTrain = transformer.fit_transform(cipinTrain)  # if-idf中的输入为已经处理过的词频矩阵
    print("(tfidfTrain.toarray()).shape : ")
    print((tfidfTrain.toarray()).shape)
    #对测试集向量化
    # cipinTest = vectorizer.transform(dataTest_list)
    # tfidfTest = transformer.transform(cipinTest)

    #用训练集训练模型
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    clf = model.fit(tfidfTrain, labelTrain_list)

    # print("对训练集进行模型评估：")
    # ytrain_pred = clf.predict(tfidfTrain)
    # print("classification_report: ")
    # print(classification_report(labelTrain_list, ytrain_pred))
    # print("\n")
    # scoreTrian = clf.score(tfidfTrain, labelTrain_list)
    # print("score:", scoreTrian)

    # print("")
    # print("对测试集进行模型评估：")
    # ytest_pred = clf.predict(tfidfTest)
    # print("classification_report: ")
    # print(classification_report(labelTest_list, ytest_pred))
    # print("\n")
    # scoreTest = clf.score(tfidfTest, labelTest_list)
    # print("score:", scoreTest)

    while 1:
        print('请输入需要预测的文本:')
        a = input()
        sentence_in = [' '.join(jieba.cut(a))]
        b = vectorizer.transform(sentence_in)
        c = transformer.transform(b)
        prd = clf.predict(c)
        # print(prd)  # ['故意伤害罪']
        print('预测类别：', prd[0])
        add = input('是否还要预测?(y/n):')
        if add == 'y':
            continue
        elif add == 'n':
            break

if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    print("Start running time: %s" % start)
  # 你运行的程序，也就是你的代码块，或者函数


    stopwords_filePath = "/code/00DataSet\\停用词.txt"
    stopWordList = getStopWord(stopwords_filePath)


    # # # 数据集的特征、标签列表
    # fileTrainPath = "B:\\code_master\\code\\00DataSet\\dataSet_50To200"
    # data_list, label_list = getDataLabelList(fileTrainPath)
    # #进行交叉验证：
    # start01 = datetime.datetime.now()
    # print("getCVResult() Start running time: %s" % start01)
    # getCVResult(data_list, label_list, stopWordList)
    # end01 = datetime.datetime.now()
    # print("getCVResult() End running time: %s" % end01)
    # print('Running time: %s Seconds' % (end01 - start01))


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

    # end = datetime.datetime.now()
    # print("End running time: %s" % end)
    # print('Running time: %s Seconds' % (end - start))




"""
运行结果：
getCVResult() Start running time: 2022-04-17 11:13:14.391413

  % sorted(inconsistent)
cv=5的分数为:  [0.60017032 0.66382798 0.67745369 0.69037479 0.65353492]
测试结果： {'fit_time': array([480.96998835, 493.49611354, 486.94839597, 467.68708158,
       438.485919  ]), 'score_time': array([105.13817143, 103.65188074, 104.23742795,  98.99656868,
        91.29974437]), 'test_precision_macro': array([0.6202474 , 0.6606913 , 0.66597   , 0.68018834, 0.6474245 ]), 'train_precision_macro': array([0.89668743, 0.89094268, 0.89241628, 0.88621638, 0.88899658]), 'test_recall_macro': array([0.59438183, 0.64861437, 0.66406398, 0.67757278, 0.64919624]), 'train_recall_macro': array([0.89174796, 0.88489888, 0.88630116, 0.88289985, 0.88433219]), 'test_f1_macro': array([0.59144559, 0.6421692 , 0.65353625, 0.66642055, 0.63621354]), 'train_f1_macro': array([0.89233013, 0.8857318 , 0.88701719, 0.88258563, 0.88478315])}
getCVResult() End running time: 2022-04-17 14:00:54.691951
Running time: 2:47:40.300538 Seconds

"""







"""
运行结果：
Building prefix dict from the default dictionary ...

Start running time: 2022-04-17 01:36:18.792775
Loading model cost 0.726 seconds.
Prefix dict has been built successfully.

  % sorted(inconsistent)
(tfidfTrain.toarray()).shape : 
(16556, 608)
对训练集进行模型评估：
classification_report: 
                precision    recall  f1-score   support

         串通投标罪       0.84      0.93      0.88       141
         交通肇事罪       0.91      0.99      0.95       141
         介绍贿赂罪       0.86      0.89      0.88       141
         代替考试罪       0.85      0.94      0.90        50
  以危险方法危害公共安全罪       0.82      0.85      0.83       141
       传授犯罪方法罪       0.88      0.92      0.90        38
         传播性病罪       0.89      0.98      0.93        50
       传播淫秽物品罪       0.90      0.94      0.92       141
           伪证罪       0.89      0.82      0.86       141
         伪造货币罪       0.96      0.98      0.97        52
           侮辱罪       0.93      0.86      0.89        63
           侵占罪       0.88      0.72      0.79       141
     侵犯公民个人信息罪       0.92      0.96      0.94       141
        侵犯著作权罪       0.96      0.96      0.96       141
         保险诈骗罪       0.93      0.98      0.96       141
        信用卡诈骗罪       0.99      0.97      0.98       141
       假冒注册商标罪       0.96      0.87      0.91       141
      偷越国（边）境罪       0.91      0.99      0.95       141
     冒充军人招摇撞骗罪       0.89      0.88      0.89       141
          刑事其他       0.69      0.59      0.64       141
      利用影响力受贿罪       0.95      0.84      0.89        49
    动植物检疫徇私舞弊罪       0.95      0.99      0.97       119
     包庇毒品犯罪分子罪       0.82      0.75      0.78        36
       协助组织卖淫罪       0.85      0.89      0.87       141
         单位受贿罪       0.85      0.87      0.86       141
         单位行贿罪       0.82      0.94      0.88       141
       危险物品肇事罪       0.91      0.72      0.81        69
         危险驾驶罪       0.99      0.99      0.99       141
           受贿罪       0.86      0.86      0.86       141
         合同诈骗罪       0.90      0.91      0.90       141
           失火罪       0.85      0.94      0.89       141
         妨害作证罪       0.89      0.85      0.87       141
      妨害信用卡管理罪       0.96      0.98      0.97       141
         妨害公务罪       0.93      0.99      0.96       141
       容留他人吸毒罪       0.92      0.93      0.93       141
        对单位行贿罪       0.90      0.67      0.77       113
   对非国家工作人员行贿罪       0.93      0.82      0.87       141
         寻衅滋事罪       0.95      0.93      0.94       141
     巨额财产来源不明罪       0.88      0.95      0.91        82
   帮助犯罪分子逃避处罚罪       0.91      0.94      0.93       105
         开设赌场罪       0.94      0.98      0.96       141
           强奸罪       0.81      0.84      0.83       141
         强迫交易罪       0.92      0.89      0.91       141
         强迫劳动罪       0.98      0.98      0.98        50
         强迫卖淫罪       0.92      0.87      0.89       119
         徇私枉法罪       0.89      0.90      0.90       141
  徇私舞弊不移交刑事案件罪       0.95      0.92      0.94        64
  扰乱无线电通讯管理秩序罪       0.97      0.99      0.98       141
       投放危险物质罪       0.87      0.87      0.87       141
           抢劫罪       0.89      0.90      0.89       141
           抢夺罪       0.82      0.91      0.86       141
         拐骗儿童罪       0.88      0.81      0.84       136
     拒不支付劳动报酬罪       0.96      1.00      0.98       141
         招摇撞骗罪       0.91      0.87      0.89       141
      持有伪造的发票罪       0.93      0.97      0.95       141
         挪用公款罪       0.88      0.83      0.85       141
       挪用特定款物罪       0.82      0.90      0.86        52
         挪用资金罪       0.85      0.81      0.83       141
           放火罪       0.85      0.78      0.81       141
         故意伤害罪       0.89      0.91      0.90       141
         故意杀人罪       0.80      0.75      0.77       141
       故意毁坏财物罪       0.89      0.88      0.88       141
         敲诈勒索罪       0.89      0.89      0.89       141
         污染环境罪       0.91      0.92      0.92       141
         滥伐林木罪       0.92      0.94      0.93       141
         滥用职权罪       0.92      0.65      0.76       141
           爆炸罪       0.88      0.68      0.77       141
         猥亵儿童罪       0.89      0.86      0.87       141
         玩忽职守罪       0.89      0.79      0.83       141
         盗伐林木罪       0.91      0.93      0.92       141
           盗窃罪       0.88      0.90      0.89       141
       破坏交通设施罪       0.92      0.86      0.89        80
     破坏易燃易爆设备罪       0.89      0.93      0.91       141
       破坏生产经营罪       0.92      0.87      0.89       141
       破坏电力设备罪       0.83      0.84      0.84       141
       破坏监管秩序罪       0.92      0.94      0.93       141
    破坏计算机信息系统罪       0.97      0.89      0.93       117
         票据诈骗罪       0.94      0.93      0.93       141
       私分国有资产罪       0.87      0.93      0.90       141
  组织他人偷越国（边）境罪       0.96      0.95      0.96       141
         组织卖淫罪       0.90      0.83      0.86       141
       组织淫秽表演罪       0.91      0.87      0.89        78
       组织考试作弊罪       0.92      0.79      0.85        43
           绑架罪       0.83      0.92      0.88       141
         职务侵占罪       0.78      0.96      0.86       141
     聚众冲击国家机关罪       0.83      0.91      0.87       131
         聚众哄抢罪       0.80      0.73      0.77        45
     聚众扰乱社会秩序罪       0.89      0.84      0.86       141
         聚众斗殴罪       0.82      0.96      0.89       141
           脱逃罪       0.83      0.79      0.81       141
         虚开发票罪       0.92      0.93      0.93       141
       虚报注册资本罪       0.87      0.85      0.86       141
           行贿罪       0.87      0.89      0.88       141
           诈骗罪       0.89      0.96      0.93       141
         诬告陷害罪       0.87      0.82      0.84       141
           诽谤罪       0.88      0.92      0.90        48
           贪污罪       0.87      0.81      0.84       141
         贷款诈骗罪       0.91      0.96      0.93       141
           赌博罪       0.90      0.91      0.90       141
         走私废物罪       0.98      0.94      0.96        68
过失以危险方法危害公共安全罪       0.91      0.88      0.89       141
     过失投放危险物质罪       0.97      0.72      0.82        39
       过失致人死亡罪       0.86      0.89      0.87       141
       过失致人重伤罪       0.82      0.91      0.86       141
  运送他人偷越国（边）境罪       0.94      0.91      0.92       141
       违法发放贷款罪       0.93      0.97      0.95       141
           逃税罪       0.94      0.94      0.94       141
           遗弃罪       0.86      0.89      0.87       134
     重大劳动安全事故罪       0.87      0.96      0.92       141
       重大责任事故罪       0.92      0.82      0.87       141
           重婚罪       0.93      0.96      0.95       141
  销售假冒注册商标的商品罪       0.89      0.95      0.92       141
         集资诈骗罪       0.91      0.92      0.92       141
    非国家工作人员受贿罪       0.90      0.85      0.87       141
     非法买卖制毒物品罪       0.96      0.96      0.96       141
       非法入侵住宅罪       0.91      0.85      0.88       141
       非法出售发票罪       0.97      0.93      0.95       141
      非法占用农用地罪       0.90      0.97      0.94       141
     非法吸收公众存款罪       0.89      0.96      0.92       141
         非法拘禁罪       0.93      0.80      0.86       141
       非法持有毒品罪       0.83      0.99      0.90       141
      非法捕捞水产品罪       0.92      0.95      0.93       141
         非法狩猎罪       0.91      0.86      0.88       141
    非法种植毒品原植物罪       0.92      0.96      0.94       141
       非法组织卖血罪       0.97      0.97      0.97       126
         非法经营罪       0.91      0.89      0.90       141
     非法获取国家秘密罪       0.92      0.99      0.96       121
         非法行医罪       0.98      0.98      0.98       141
     非法进行节育手术罪       0.92      0.96      0.94       141
         非法采矿罪       0.96      0.96      0.96       141
         高利转贷罪       0.92      0.95      0.94        38

      accuracy                           0.90     16556
     macro avg       0.90      0.89      0.89     16556
  weighted avg       0.90      0.90      0.90     16556



score: 0.8962309736651365

对测试集进行模型评估：
classification_report: 
                precision    recall  f1-score   support

         串通投标罪       0.72      0.83      0.77        59
         交通肇事罪       0.59      0.64      0.62        59
         介绍贿赂罪       0.55      0.80      0.65        59
         代替考试罪       0.62      0.76      0.68        21
  以危险方法危害公共安全罪       0.52      0.51      0.51        59
       传授犯罪方法罪       0.55      0.73      0.63        15
         传播性病罪       0.73      0.95      0.83        20
       传播淫秽物品罪       0.66      0.73      0.69        59
           伪证罪       0.53      0.58      0.55        59
         伪造货币罪       0.75      0.82      0.78        22
           侮辱罪       0.46      0.50      0.48        26
           侵占罪       0.61      0.64      0.63        59
     侵犯公民个人信息罪       0.84      0.83      0.84        59
        侵犯著作权罪       0.83      0.66      0.74        59
         保险诈骗罪       0.68      0.88      0.77        59
        信用卡诈骗罪       0.98      0.83      0.90        59
       假冒注册商标罪       0.86      0.75      0.80        59
      偷越国（边）境罪       0.78      0.97      0.86        59
     冒充军人招摇撞骗罪       0.55      0.76      0.64        59
          刑事其他       0.09      0.15      0.11        59
      利用影响力受贿罪       0.60      0.75      0.67        20
    动植物检疫徇私舞弊罪       0.81      0.94      0.87        50
     包庇毒品犯罪分子罪       0.38      0.33      0.36        15
       协助组织卖淫罪       0.60      0.71      0.65        59
         单位受贿罪       0.72      0.81      0.76        59
         单位行贿罪       0.59      0.63      0.61        59
       危险物品肇事罪       0.38      0.10      0.16        29
         危险驾驶罪       0.56      0.34      0.42        59
           受贿罪       0.62      0.47      0.54        59
         合同诈骗罪       0.65      0.54      0.59        59
           失火罪       0.54      0.75      0.63        59
         妨害作证罪       0.52      0.49      0.50        59
      妨害信用卡管理罪       0.75      0.76      0.76        59
         妨害公务罪       0.86      0.92      0.89        59
       容留他人吸毒罪       0.84      0.63      0.72        59
        对单位行贿罪       0.71      0.32      0.44        47
   对非国家工作人员行贿罪       0.58      0.44      0.50        59
         寻衅滋事罪       0.67      0.51      0.58        59
     巨额财产来源不明罪       0.51      0.66      0.57        35
   帮助犯罪分子逃避处罚罪       0.65      0.82      0.73        44
         开设赌场罪       0.81      0.75      0.78        59
           强奸罪       0.50      0.56      0.53        59
         强迫交易罪       0.63      0.71      0.67        59
         强迫劳动罪       0.76      0.76      0.76        21
         强迫卖淫罪       0.65      0.78      0.71        50
         徇私枉法罪       0.67      0.58      0.62        59
  徇私舞弊不移交刑事案件罪       0.64      0.67      0.65        27
  扰乱无线电通讯管理秩序罪       0.78      0.66      0.72        59
       投放危险物质罪       0.63      0.78      0.70        59
           抢劫罪       0.50      0.25      0.34        59
           抢夺罪       0.63      0.80      0.70        59
         拐骗儿童罪       0.50      0.53      0.51        57
     拒不支付劳动报酬罪       0.79      0.98      0.88        59
         招摇撞骗罪       0.53      0.29      0.37        59
      持有伪造的发票罪       0.88      0.88      0.88        59
         挪用公款罪       0.57      0.39      0.46        59
       挪用特定款物罪       0.23      0.29      0.26        21
         挪用资金罪       0.46      0.47      0.47        59
           放火罪       0.38      0.31      0.34        59
         故意伤害罪       0.91      0.69      0.79        59
         故意杀人罪       0.55      0.27      0.36        59
       故意毁坏财物罪       0.44      0.31      0.36        59
         敲诈勒索罪       0.42      0.19      0.26        59
         污染环境罪       0.48      0.24      0.32        59
         滥伐林木罪       0.74      0.85      0.79        59
         滥用职权罪       0.50      0.24      0.32        59
           爆炸罪       0.39      0.25      0.31        59
         猥亵儿童罪       0.32      0.29      0.30        59
         玩忽职守罪       0.55      0.41      0.47        59
         盗伐林木罪       0.62      0.78      0.69        59
           盗窃罪       0.89      0.98      0.94        59
       破坏交通设施罪       0.77      0.59      0.67        34
     破坏易燃易爆设备罪       0.62      0.81      0.71        59
       破坏生产经营罪       0.68      0.61      0.64        59
       破坏电力设备罪       0.62      0.61      0.62        59
       破坏监管秩序罪       0.65      0.86      0.74        59
    破坏计算机信息系统罪       0.59      0.55      0.57        49
         票据诈骗罪       0.78      0.83      0.80        59
       私分国有资产罪       0.53      0.68      0.60        59
  组织他人偷越国（边）境罪       0.79      0.64      0.71        59
         组织卖淫罪       0.70      0.44      0.54        59
       组织淫秽表演罪       0.54      0.45      0.49        33
       组织考试作弊罪       0.33      0.11      0.17        18
           绑架罪       0.41      0.66      0.50        59
         职务侵占罪       0.83      0.83      0.83        59
     聚众冲击国家机关罪       0.56      0.77      0.65        56
         聚众哄抢罪       0.43      0.33      0.38        18
     聚众扰乱社会秩序罪       0.63      0.64      0.64        59
         聚众斗殴罪       0.80      0.93      0.86        59
           脱逃罪       0.51      0.59      0.55        59
         虚开发票罪       0.75      0.73      0.74        59
       虚报注册资本罪       0.67      0.80      0.73        59
           行贿罪       0.48      0.25      0.33        59
           诈骗罪       0.89      0.71      0.79        59
         诬告陷害罪       0.53      0.53      0.53        59
           诽谤罪       0.50      0.50      0.50        20
           贪污罪       0.37      0.19      0.25        59
         贷款诈骗罪       0.75      0.92      0.82        59
           赌博罪       0.72      0.69      0.71        59
         走私废物罪       0.58      0.75      0.66        28
过失以危险方法危害公共安全罪       0.45      0.59      0.51        59
     过失投放危险物质罪       0.38      0.56      0.45        16
       过失致人死亡罪       0.48      0.51      0.49        59
       过失致人重伤罪       0.59      0.86      0.70        59
  运送他人偷越国（边）境罪       0.71      0.86      0.78        59
       违法发放贷款罪       0.66      0.93      0.77        59
           逃税罪       0.69      0.90      0.78        59
           遗弃罪       0.56      0.64      0.60        56
     重大劳动安全事故罪       0.60      0.81      0.69        59
       重大责任事故罪       0.69      0.46      0.55        59
           重婚罪       0.78      0.92      0.84        59
  销售假冒注册商标的商品罪       0.83      0.97      0.89        59
         集资诈骗罪       0.70      0.71      0.71        59
    非国家工作人员受贿罪       0.68      0.78      0.72        59
     非法买卖制毒物品罪       0.52      0.69      0.59        59
       非法入侵住宅罪       0.28      0.15      0.20        59
       非法出售发票罪       0.87      0.92      0.89        59
      非法占用农用地罪       0.57      0.49      0.53        59
     非法吸收公众存款罪       0.75      0.76      0.76        59
         非法拘禁罪       0.33      0.17      0.22        59
       非法持有毒品罪       0.64      0.90      0.75        59
      非法捕捞水产品罪       0.83      0.76      0.80        59
         非法狩猎罪       0.56      0.42      0.48        59
    非法种植毒品原植物罪       0.65      0.92      0.76        59
       非法组织卖血罪       0.37      0.28      0.32        53
         非法经营罪       0.72      0.22      0.34        59
     非法获取国家秘密罪       0.67      0.73      0.70        51
         非法行医罪       0.84      0.73      0.78        59
     非法进行节育手术罪       0.63      0.75      0.68        59
         非法采矿罪       0.82      0.83      0.82        59
         高利转贷罪       0.75      0.38      0.50        16

      accuracy                           0.63      6927
     macro avg       0.62      0.62      0.61      6927
  weighted avg       0.63      0.63      0.62      6927



score: 0.6321639959578461
End running time: 2022-04-17 01:56:28.880659
Running time: 0:20:10.087884 Seconds

Process finished with exit code 0


"""


#



























