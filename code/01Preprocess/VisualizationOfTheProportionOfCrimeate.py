# -*- coding:utf-8 -*-

import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot

df = pd.read_csv("B:\\code_master\\code\\01Preprocess\\ChargeNum.csv")
df['rate'] = df['num'].map(lambda i : i/(df['num'].sum()))
# df.to_csv('B:\\code_master\\code\\01Preprocess\\ChargeNumWithRate.csv',encoding="utf_8_sig")
data_a = [round(n*100,4) for n in df['rate']]

bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add_xaxis(df['name'].values.tolist())
            .add_yaxis("罪名数量占比",
                       data_a,
                       label_opts = opts.LabelOpts(is_show=False)
                       # label_opts=opts.LabelOpts(formatter="{c} %"),
                       )
            .set_colors(['red'])
            .set_global_opts(
            title_opts=opts.TitleOpts(title="案件罪名数量占比分布"),
            xaxis_opts=opts.AxisOpts(name_rotate=60,axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name='案件占比',
                                     name_location='middle',
                                     name_gap=40,
                                     axislabel_opts=opts.LabelOpts(formatter="{value} %"),
                                     interval=10)
        )
    )
# bar.render()
make_snapshot(snapshot, bar.render(),"ChargeNumOfProportion.png")



