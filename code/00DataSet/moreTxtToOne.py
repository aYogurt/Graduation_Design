# -*- coding:utf-8 -*-


from os import listdir
import os
# inputPath = "B:\\code_master\\code\\00DataSet\\dataTestSet_2To50"
inputPath = "B:\\code_master\\code\\00DataSet\\dataTrainSet_2To50"
fileName = listdir(inputPath)

# dataFilePath = "B:\\code_master\\code\\00DataSet\\test"
# if not os.path.isdir(dataFilePath):
#     os.makedirs(dataFilePath)


for i in range(len(fileName)):
    each = inputPath + '\\' + fileName[i]
    f1 = open(each, 'r', encoding="utf-8")
    cont = f1.readlines()
    print(fileName[i])
    for i in range(len(cont)):
        with open('B:\\code_master\\code\\00DataSet\\' + '其他类.txt', 'a', encoding="utf-8") as f2:
            f2.write(cont[i])
            f2.close()
            # print(cont[i])
    print(len(cont))
    print("")
    f1.close()





