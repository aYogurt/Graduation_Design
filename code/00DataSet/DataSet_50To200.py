# -*- coding:utf-8 -*-

import  json
from os import listdir
import os
import re
dataFilePath = "B:\\code_master\\code\\00DataSet\\dataSet_50To200"
if not os.path.isdir(dataFilePath):
    os.makedirs(dataFilePath)

FilePath = "G:\\data_Under200"
filename = listdir(FilePath)
for i in range(len(filename)):
    f2 = open(FilePath +"\\" +filename[i], 'r', encoding="utf-8")
    cont = f2.readlines()
    print(len(cont))
    print(filename[i])
    if len(cont) <= 200 and len(cont) >= 50:
        for j in range(len(cont)):
            f1 = open(dataFilePath + "\\" + filename[i], 'a', encoding="utf-8")
            f1.write(cont[j])
            f1.close()
    else:
        f2.close()
    f2.close()


