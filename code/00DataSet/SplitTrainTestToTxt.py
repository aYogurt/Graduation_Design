# -*- coding:utf-8 -*-

import os
from os import listdir



# TrainFilePath = "G:\\dataTrainSet_50To200"
# TestFilePath = "G:\\dataTestSet_50To200"
# TrainFilePath = "dataTrainSet_50To200"
# TestFilePath = "dataTestSet_50To200"
TrainFilePath = "dataTrainSet_2To50"
TestFilePath = "dataTestSet_2To50"
if not os.path.isdir(TrainFilePath):
    os.makedirs(TrainFilePath)
if not os.path.isdir(TestFilePath):
    os.makedirs(TestFilePath)


# FilePath = "G:\\dataSet180_200"
# FilePath = "dataSet_50To200"
FilePath = "dataSet_2To50"
filename = listdir(FilePath)
for i in range(len(filename)):
    with open(FilePath +"\\" +filename[i] ,'r',encoding="utf-8") as f1:
        # print(f1)
        print(filename[i])
        count = f1.readlines()
        # print(type(count))
        # print(len(count))
        fileNum = len(count) #一个txt文件里的文本数量（即样本量），一行文本即一个样本量
        print(fileNum)
        splitTrain_num = 0.7 #划分训练集的概率
        train_num = int(fileNum*splitTrain_num) #int()向下取整，例如print(int(14.12)) 结果：14
        print(train_num)
        print("\n")
        for j in range(len(count)):
            if j <= train_num-1:
                f2 = open(TrainFilePath+"\\" +filename[i], 'a', encoding="utf-8")
                f2.write(count[j])
                f2.close()
            elif j >train_num-1:
            # else:
                f3 = open(TestFilePath+"\\" +filename[i], 'a', encoding="utf-8")
                f3.write(count[j])
                f3.close()

        f1.close()




















