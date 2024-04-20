# -*- coding:utf-8 -*-

import  json
from os import listdir
import os
import re
'''
把包含.json文件的文件夹名放进列表，
并做好从小到大的排列（文件夹名是数字形式）。
'''
firstPath = "G:\\itslaw"
firstLists = listdir(firstPath)#主目录名称
# print(firstLists)
for each1 in firstLists:
    eachFirstPath = firstPath + "\\" + each1#主目录路径
    # print(eachFirstPath)  #G:\itslaw\2014_2_1_1865
    secondLists = listdir(eachFirstPath)
    # print(secondLists)
    for each2 in secondLists:
        eachSecondPath = eachFirstPath + "\\" + each2#次目录路径
        # print(eachSecondPath)  #G:\itslaw\2014_2_1_1865\2_1
        thirdLists = listdir(eachSecondPath)
        # print(thirdLists)
        #对thirdLists排序：
        number = list(map(int, thirdLists))
        number = sorted(number,reverse= False)
        thirdLists = list(map(str, number))
        # print(thirdLists)
        for each3 in thirdLists:
            eachThirdPath = eachSecondPath + "\\" + each3
            # print(eachThirdPath)
            """
            print(eachThirdPath)结果：
            G:\itslaw\2014_2_1_1865\2_1\1
            G:\itslaw\2014_2_1_1865\2_1\2
            ...
            """
            json_files = [pos_json for pos_json in os.listdir(eachThirdPath) if not pos_json.endswith('.txt')]
            # print(json_files)
            """
            print(json_files)结果：
            ['0']
            ['0', '1']
            ['0', '1']
            ...
            """
            if len(json_files) == 0:
                continue
            # result = []  # 接收json数据
            for file in json_files:
                # print(file)
                with open(eachThirdPath + "\\" + file, 'r', encoding="utf-8") as f:
                    li = []
                    li_text = []
                    try:
                        for lines in f:
                            test = json.loads(lines)
                            if test['result']['message'] == "成功":  # 去除没成功录入信息的

                                try:
                                    casename = test['data']['fullJudgement']['reason']['name']
                                except:
                                    continue

                                if casename.find("、") == -1:  # 去除多标签罪名
                                    # li.append(casename)
                                    try:
                                        number = len(test['data']['fullJudgement']['paragraphs'][2]['subParagraphs'])
                                    except:
                                        print(test)
                                        continue

                                    if number == 1:
                                        context = ''
                                        context = test['data']['fullJudgement']['paragraphs'][2]['subParagraphs'][0]['text']
                                    elif number > 1:
                                        context = ''
                                        for i in range(0, number):
                                            context += (test['data']['fullJudgement']['paragraphs'][2]['subParagraphs'][i]['text'])
                                    else:
                                        print("为空！")
                                    # li.append(casename)
                                    # li_text.append(context)
                                else:
                                    continue

                            else:
                                continue

                            li.append(casename)
                            li_text.append(context)
                    except:
                        print(f)  # <_io.TextIOWrapper name='G:\\itslaw\\2014_2_1_1865\\2_1\\1322\\0' mode='r' encoding='utf-8'>

                # pattern = r"(依照|本院依据?|依据<a?|根据?|<a?).*(规定)"
                pattern = r"(<a).*(</a>)"
                result = []
                for i in range(len(li_text)):
                    result.append(re.sub(pattern, '', li_text[i]))

                for i in range(len(li)):
                    # print(li[i]+":"+li_text[i])
                    filename = 'G:\\data3\\' + li[i] + '.txt'
                    with open(filename, 'a', encoding="utf-8") as f:
                        f.write(result[i])
                        f.write("\n")



# data 所有单重、多重罪名 txt文件
#
# data3
















            #             # result.append(json.loads(lines))
            # # print(eachThirdPath + ":")
            # # print(len(result))
            # """
            # print(eachThirdPath + ":")
            # print(len(result))
            # 结果如下：
            # G:\itslaw\2014_2_1_1865\2_1\1:
            # 2
            # G:\itslaw\2014_2_1_1865\2_1\2:
            # 175
            # ...
            # """



