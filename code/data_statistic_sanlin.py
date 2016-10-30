# -*- encoding:utf-8 -*-

import re
import sys
import time
import fileinput
import numpy as np

from utils import *
from constants import *

if len(sys.argv) <= 1:
	raise Exception('parameter wrong.')

if sys.argv[1].startswith('plot'):
	import matplotlib.pyplot as plt
	from pylab import *
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	mpl.rcParams['axes.unicode_minus'] = False


def statistic_by_space(filename):
	field_dict = {}; geo = [[0 for _ in range(YNUM)] for _ in range(XNUM)]
	for line in fileinput.input(filename):
		if fileinput.lineno() == 1:
			field_dict = {field:i for i, field in enumerate(line.strip().decode('utf-8').split(u'\t'))}
		else:
			fields = line.strip().decode('utf-8').split(u'\t')
			COORDX,COORDY = map(lambda x:float(x),[fields[field_dict[field]] for field in [u'坐标X',u'坐标Y']])
			cx,cy = int((float(COORDX)-COORDX_min)/(COORDX_max-COORDX_min)*XNUM), int((float(COORDY)-COORDY_min)/(COORDY_max-COORDY_min)*YNUM)
			geo[cx][cy] = 1
	fileinput.close()

	plt.figure(figsize=(5,5))
	(X, Y), C = meshgrid(np.arange(XNUM), np.arange(YNUM)), np.array(geo)
	cset = pcolormesh(X, Y, C.T, cmap=cm.get_cmap("OrRd"))
	plt.show()


if __name__ == '__main__':
	if sys.argv[1] == 'plot_statistic_by_space':
		statistic_by_space('../data/sanlin/parts/{0}.txt'.format(sys.argv[2])) # 部件门牌

