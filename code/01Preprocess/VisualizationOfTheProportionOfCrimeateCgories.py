# -*- coding:utf-8 -*-

import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot

df = pd.read_csv("B:\\code_master\\code\\01Preprocess\\ALLChargeNum.csv")
# df.iloc[:,1]
# df["name"].values.tolist()
# df["num"].values.tolist()
list1 = df["name"].values.tolist()
list2 = df["num"].values.tolist()
dict = {}
for i in range(0,len(list1)):
    # print((list1[i]).split('、'))
    key = len((list1[i]).split('、'))
    # print(key)
    num = list2[i]
    dict[key] = dict[key] +num if key in dict else num
    # print(num)
    # dict[key] = dict.get(key,0) + num
print(dict)
#{1: 1998603, 4: 158427, 3: 71415, 2: 41343, 7: 6513, 5: 8678, 6: 1220}

#将字典dict转为dataframe
df = pd.DataFrame.from_dict(dict, orient='index',columns=['num'])
df = df.reset_index().rename(columns = {'index':'classNum'})
#在dataframe里添加一列数据
df['rate'] = df['num'].map(lambda i : i/(df['num'].sum()))
#根据列“classNum”排序
sorted_df = df.sort_values(by = 'classNum', ascending = True)
# sorted_df.reset_index(drop=True).to_csv('B:\\code_master\\code\\01Preprocess\\AllClassNum.csv',encoding="utf_8_sig")
data_a = [round(n*100,2) for n in sorted_df['rate']]

bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add_xaxis(sorted_df['classNum'].values.tolist())
            .add_yaxis("罪名个数类别", data_a, label_opts=opts.LabelOpts(formatter="{c} %"))
            .set_global_opts(title_opts=opts.TitleOpts(title="案件罪名个数类别占比分布"),
                            xaxis_opts=opts.AxisOpts(name='案件罪名标签数量',
                                                     name_location='middle',
                                                     name_gap=20),
                            yaxis_opts=opts.AxisOpts(name='案件占比',
                                                     name_location='middle',
                                                     name_gap=40,
                                                     axislabel_opts=opts.LabelOpts(formatter="{value} %"), interval=10))
    )
# bar.render()
make_snapshot(snapshot, bar.render(),"chargesNumDistribution.png")
#

    # print(type(list1[i]))
# print(number)
# df['rate'] = df.apply(lambda i : i/(df['num'].sum()), axis = 1)#error
# df['rate'] = df['num'].map(lambda i : i/(df['num'].sum())) #right

# data_a = [round(n*100,2) for n in sorted_df['rate']]
