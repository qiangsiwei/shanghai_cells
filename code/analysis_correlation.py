# -*- encoding:utf-8 -*-

import re
import sys
import json
import time
import fileinput
import numpy as np

from utils import *
from constants import *

if len(sys.argv) != 2:
	raise Exception('parameter wrong.')

if sys.argv[1].startswith('plot'):
	import matplotlib.pyplot as plt
	from pylab import *
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	mpl.rcParams['axes.unicode_minus'] = False


def generate_tensor(filename, time_granularity='%W'):
	'''
		Generate tensor.
	'''
	# TBD: use collections.defaultdict()
	case_dict = {}
	
	for line in fileinput.input(filename):
		if fileinput.lineno() % 10**4 == 0: sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
		TASKID, COORDX, COORDY, INFOSOURCENAME, DISCOVERTIME, SOLVINGTIME, \
		ADDRESS, STREETNAME, DESCRIPTION, ENDRESULT, URGENTDEGREE, USEREVALUATE, \
		INFOBCNAME, INFOSCNAME, INFOZCNAME, CASEENDBCNAME, CASEENDSCNAME = map(lambda x:x.strip(), line.decode('utf-8').split(u'\t'))
		INFOBCNAME, INFOSCNAME, INFOZCNAME = map(lambda x:re.sub(ur'\(浦东\)','',x), [INFOBCNAME, INFOSCNAME, INFOZCNAME])
		if COORDX and COORDY and float(COORDX) and float(COORDY) and DISCOVERTIME:
			if time_granularity == '%w':
				timeslot = int(time.strftime(time_granularity,time.strptime(DISCOVERTIME,'%Y/%m/%d %H:%M:%S')))
			if time_granularity == '%W':
				timeslot = int(time.strftime(time_granularity,time.strptime(DISCOVERTIME,'%Y/%m/%d %H:%M:%S'))) + (53 if DISCOVERTIME.startswith('2016') else 0)
			if time_granularity == '%j':
				timeslot = int(time.strftime(time_granularity,time.strptime(DISCOVERTIME,'%Y/%m/%d %H:%M:%S'))) + (366 if DISCOVERTIME.startswith('2016') else 0)
			cx,cy = int((float(COORDX)-COORDX_min)/(COORDX_max-COORDX_min)*XNUM), int((float(COORDY)-COORDY_min)/(COORDY_max-COORDY_min)*YNUM)
			# update case_dict
			if INFOBCNAME: 
				grid = '{0},{1}'.format(cx,cy)
				case_dict[INFOBCNAME] = case_dict.get(INFOBCNAME,{})
				case_dict[INFOBCNAME][grid] = case_dict[INFOBCNAME].get(grid,[0 for _ in range(NT)])
				case_dict[INFOBCNAME][grid][timeslot] += 1
	fileinput.close()

	with open('./tensor/tensor.json','w') as outfile:
		outfile.write(json.dumps(case_dict))


def time_autocorrelation():
	'''
		Compute time autocorrelation.
	'''
	from statsmodels.tsa.stattools import acf

	case_dict = json.loads(open('./tensor/tensor.json').read())
	for _plotname, _colnum, _category in [('12315市民服务热线管理',3,category1),('事部件词语',4,category2)]:
		plt.figure(figsize=(12,12))
		subplots_adjust(left=0.06,right=0.98,hspace=0.4,wspace=0.4)
		for i, name in enumerate(_category):
			subplot(_colnum,_colnum,i+1)
			timelines = sorted([(grid, timeline) for grid, timeline in case_dict[name].iteritems()],key=lambda x:sum(x[1]),reverse=True)
			m = (np.array([acf(timeline)*sum(timeline) for grid, timeline in timelines]).sum(axis=0)/np.array([sum(timeline) for grid, timeline in timelines]).sum())[1:]
			plt.plot(range(len(m)), m, 'ko-', markeredgecolor='k', linewidth=2)
			plt.xlabel(u'天'); plt.ylabel(u'自相关系数')
			for i in xrange(len(m)):
				if i%7==6: plt.plot([i,i], [min(m),max(m)], 'r--', linewidth=1)
			plt.title(name)
		plt.savefig('./plots/time_autocorrelation_{0}.png'.format(_plotname))


def space_crosscorrelation():
	'''
		Compute space crosscorrelation.
	'''
	from numpy import corrcoef

	case_dict = json.loads(open('./tensor/tensor.json').read())
	for _plotname, _colnum, _category in [('12315市民服务热线管理',3,category1),('事部件词语',4,category2)]:
		plt.figure(figsize=(12,12))
		subplots_adjust(left=0.06,right=0.98,hspace=0.4,wspace=0.4)
		for i, name in enumerate(_category):
			print name
			subplot(_colnum,_colnum,i+1)
			timelines = sorted([(grid, timeline) for grid, timeline in case_dict[name].iteritems()],key=lambda x:sum(x[1]),reverse=True)[:200]
			dist_corrs = {}
			for i in xrange(len(timelines)-1):
				print '{0}/{1}'.format(i,len(timelines))
				for j in xrange(i+1,len(timelines)):
					weight = sum(timelines[i][1])*sum(timelines[j][1])
					corr = corrcoef(timelines[i][1], timelines[j][1])[0][1]*weight
					dist = ((np.array(map(lambda x:int(x),timelines[i][0].split(',')))-np.array(map(lambda x:int(x),timelines[j][0].split(','))))**2).sum()**0.5
					dist = int(dist/2)*2 if dist < 10 else int(dist/4)*4
					if dist < 50: dist_corrs[(COORDX_max-COORDX_min)/XNUM*dist] = dist_corrs.get((COORDX_max-COORDX_min)/XNUM*dist,[])+[[corr,weight]]
			x, y = zip(*sorted([(dist,np.array(corrs)[:,0].sum()/np.array(corrs)[:,1].sum()) for dist,corrs in dist_corrs.iteritems()],key=lambda x:x[0]))
			ax = plt.gca()
			ax.xaxis.get_major_formatter().set_powerlimits((0,1))
			plt.plot(x, y, 'ko-', markeredgecolor='k', linewidth=2)
			plt.xlabel(u'距离'); plt.ylabel(u'互相关系数')
			if min(y)<0:plt.plot([min(x),max(x)], [0,0], 'r--', linewidth=1)
			plt.title(name)
		plt.savefig('./plots/space_crosscorrelation_{0}.png'.format(_plotname))


if __name__ == '__main__':
	if sys.argv[1] == 'generate_tensor':
		NT = NT_DAYS; generate_tensor('../data/extracted.tsv',time_granularity='%j')
	if sys.argv[1] == 'plot_time_autocorrelation':
		time_autocorrelation()
	if sys.argv[1] == 'plot_space_crosscorrelation':
		space_crosscorrelation()

