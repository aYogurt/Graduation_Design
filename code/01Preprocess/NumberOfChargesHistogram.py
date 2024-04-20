# -*- coding:utf-8 -*-

import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot

df = pd.read_csv("B:\\code_master\\code\\01Preprocess\\ChargeNum.csv")
df.iloc[:,1]
df["name"].values.tolist()
df["num"].values.tolist()

bar = (
    Bar()
        .add_xaxis(df["name"].values.tolist())
        .add_yaxis("刑事犯罪罪名", df["num"].values.tolist(),label_opts = opts.LabelOpts(is_show=False))
        .set_colors(['red'])
        .set_global_opts(
        xaxis_opts=opts.AxisOpts(name_rotate=60,axislabel_opts=opts.LabelOpts(rotate=45)),
        yaxis_opts=opts.AxisOpts(name='案件数量',
                                 name_location='middle',
                                 name_gap=55),
        title_opts=opts.TitleOpts(title="案件罪名数量分布"),
        # datazoom_opts=opts.DataZoomOpts(pos_top= 20),
        )
    )
# bar.render()
make_snapshot(snapshot, bar.render(),"ChargeNum.png")
# make_snapshot(snapshot, bar.render(),"ChargeNumWithZoom.png")

