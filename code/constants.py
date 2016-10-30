# -*- encoding:utf-8 -*-

COORDX_min, COORDX_max, COORDY_min, COORDY_max = -50000, 75000, -50000, 50000

XNUM, YNUM = 100, 100
NT_INWK = 7
NT_WEEK = 100
NT_DAYS = 700

order1re = r'^[^\t]+'; order2re = r'^\t[^\t]+'; order3re = r'^\t\t[^\t]+'
# 12315市民服务热线管理（9）
category1 = [u'社会管理类',u'建设交通类',u'经济综合类',u'科教文卫类',u'公安政法类',u'公用事业类',u'社会团体类',u'安全监管类',u'其他类']
# 事部件词语（18）
category2 = [u'公用设施',u'道路交通',u'环卫环保',u'园林绿化',u'其它设施',u'环卫市容',u'设施管理',u'突发事件',u'街面秩序',u'市场监管',u'小区管理',u'农村管理',u'街面治安',u'地下空间',u'专项类',u'其他事件']
