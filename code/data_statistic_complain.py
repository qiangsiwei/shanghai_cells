# -*- encoding:utf-8 -*-

import re
import sys
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


def geo_coordinates_compare(filename='./test/geo_coordinates.txt'):
	coordinates = np.array([map(lambda x:float(x),line.strip().split(',')) for line in fileinput.input(filename)])
	grids, geos = coordinates[:,:2], coordinates[:,2:]
	plt.figure(figsize=(12,5))
	subplots_adjust(left=0.10,right=0.95,hspace=0.4,wspace=0.4)
	subplot(1,2,1); plt.scatter(grids[:,0], grids[:,1], c=['r','b','g','k'], alpha=1.); plt.title(u'原始坐标系')
	subplot(1,2,2); plt.scatter(geos[:,1], geos[:,0], c=['r','b','g','k'], alpha=1.); plt.title(u'经纬度坐标系')
	plt.show()


def statistic(filename, time_granularity='%W'):
	# TBD: use collections.defaultdict()
	case_dict = {}
	# caseend_dict = {}
	
	for line in fileinput.input(filename):
		if fileinput.lineno() % 10**4 == 0: print sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
		TASKID, COORDX, COORDY, INFOSOURCENAME, DISCOVERTIME, SOLVINGTIME, \
		ADDRESS, STREETNAME, DESCRIPTION, EXECUTEDEPTNAME, URGENTDEGREE, USEREVALUATE, \
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
				case_dict[INFOBCNAME] = case_dict.get(INFOBCNAME,{'timeline':[0]*NT,\
																  'geo':[[0 for y in range(YNUM)] for x in range(XNUM)],\
																  'dict':{}})
				case_dict[INFOBCNAME]['timeline'][timeslot] += 1
				case_dict[INFOBCNAME]['geo'][cx][cy] += 1
			if INFOSCNAME: 
				case_dict[INFOBCNAME]['dict'][INFOSCNAME] = case_dict[INFOBCNAME]['dict'].get(INFOSCNAME,{'timeline':[0]*NT,\
																										  'geo':[[0 for y in range(YNUM)] for x in range(XNUM)],\
																										  'dict':{}})
				case_dict[INFOBCNAME]['dict'][INFOSCNAME]['timeline'][timeslot] += 1
				case_dict[INFOBCNAME]['dict'][INFOSCNAME]['geo'][cx][cy] += 1
			if INFOZCNAME: 
				case_dict[INFOBCNAME]['dict'][INFOSCNAME]['dict'][INFOZCNAME] = case_dict[INFOBCNAME]['dict'][INFOSCNAME]['dict'].get(INFOZCNAME,{'timeline':[0]*NT,\
																																				  'geo':[[0 for y in range(YNUM)] for x in range(XNUM)]})
				case_dict[INFOBCNAME]['dict'][INFOSCNAME]['dict'][INFOZCNAME]['timeline'][timeslot] += 1
				case_dict[INFOBCNAME]['dict'][INFOSCNAME]['dict'][INFOZCNAME]['geo'][cx][cy] += 1
			
			# # update caseend_dict
			# if CASEENDBCNAME:
			# 	caseend_dict[CASEENDBCNAME] = caseend_dict.get(CASEENDBCNAME,{'timeline':[0]*NT,\
			# 																  'geo':[[0 for y in range(YNUM)] for x in range(XNUM)],\
			# 																  'dict':{}})
			# 	caseend_dict[CASEENDBCNAME]['timeline'][timeslot] += 1
			# 	caseend_dict[CASEENDBCNAME]['geo'][cx][cy] += 1
			# if CASEENDSCNAME:
			# 	caseend_dict[CASEENDBCNAME]['dict'][CASEENDSCNAME] = caseend_dict[CASEENDBCNAME]['dict'].get(CASEENDSCNAME,{'timeline':[0]*NT,\
			# 																												'geo':[[0 for y in range(YNUM)] for x in range(XNUM)]})
			# 	caseend_dict[CASEENDBCNAME]['dict'][CASEENDSCNAME]['timeline'][timeslot] += 1
			# 	caseend_dict[CASEENDBCNAME]['dict'][CASEENDSCNAME]['geo'][cx][cy] += 1
	
	fileinput.close()

	timelineformat = lambda timeline:','.join(map(lambda cnt:str(cnt),timeline)).decode('utf-8')
	geoformat = lambda geo:','.join([str(geo[x][y]) for x in range(XNUM) for y in range(YNUM)]).decode('utf-8')
	
	with open('./statistic/CASE_HIERARCHY.txt','w') as outfile:
		# for BCNAME in sorted(case_dict):
		for BCNAME, _ in sorted(case_dict.iteritems(),key=lambda x:sum(x[1]['timeline']),reverse=True):
			outfile.write(u'{0}\t{1}\t{2}\t{3}\n'.format(BCNAME,\
														sum(case_dict[BCNAME]['timeline']),\
														timelineformat(case_dict[BCNAME]['timeline']),\
														geoformat(case_dict[BCNAME]['geo'])
														).encode('utf-8'))
			# for SCNAME in sorted(case_dict[BCNAME]['dict']):
			for SCNAME, _ in sorted(case_dict[BCNAME]['dict'].iteritems(),key=lambda x:sum(x[1]['timeline']),reverse=True):
				outfile.write(u'\t{0}\t{1}\t{2}\t{3}\n'.format(SCNAME,\
															  sum(case_dict[BCNAME]['dict'][SCNAME]['timeline']),\
															  timelineformat(case_dict[BCNAME]['dict'][SCNAME]['timeline']),\
															  geoformat(case_dict[BCNAME]['dict'][SCNAME]['geo'])
															  ).encode('utf-8'))
				# for ZCNAME in sorted(case_dict[BCNAME]['dict'][SCNAME]['dict']):
				for ZCNAME, _ in sorted(case_dict[BCNAME]['dict'][SCNAME]['dict'].iteritems(),key=lambda x:sum(x[1]['timeline']),reverse=True):
					outfile.write(u'\t\t{0}\t{1}\t{2}\t{3}\n'.format(ZCNAME,\
																	sum(case_dict[BCNAME]['dict'][SCNAME]['dict'][ZCNAME]['timeline']),\
																	timelineformat(case_dict[BCNAME]['dict'][SCNAME]['dict'][ZCNAME]['timeline']),\
																	geoformat(case_dict[BCNAME]['dict'][SCNAME]['dict'][ZCNAME]['geo'])
																	).encode('utf-8'))

	# with open('./statistic/CASEEND_HIERARCHY.txt','w') as outfile:
	# 	# for BCNAME in sorted(caseend_dict):
	# 	for BCNAME, _ in sorted(caseend_dict.iteritems(),key=lambda x:sum(x[1]['timeline']),reverse=True):
	# 		outfile.write(u'{0}\t{1}\t{2}\t{3}\n'.format(BCNAME,\
	# 													sum(caseend_dict[BCNAME]['timeline']),\
	# 													timelineformat(caseend_dict[BCNAME]['timeline']),\
	# 													geoformat(caseend_dict[BCNAME]['geo'])
	# 													).encode('utf-8'))
	# 		# for SCNAME in sorted(caseend_dict[BCNAME]['dict']):
	# 		for SCNAME, _ in sorted(caseend_dict[BCNAME]['dict'].iteritems(),key=lambda x:sum(x[1]['timeline']),reverse=True):
	# 			outfile.write(u'\t{0}\t{1}\t{2}\t{3}\n'.format(SCNAME,\
	# 														  sum(caseend_dict[BCNAME]['dict'][SCNAME]['timeline']),\
	# 														  timelineformat(caseend_dict[BCNAME]['dict'][SCNAME]['timeline']),\
	# 														  geoformat(caseend_dict[BCNAME]['dict'][SCNAME]['geo'])
	# 														  ).encode('utf-8'))


def statistic_simplify(filename, show_coverage=False):
	covered, uncovered = 0, 0; byBCNAME = {}
	with open(re.sub('.txt','_STATISTIC.txt',filename),'w') as outfile:
		for line in fileinput.input(filename):
			_name, _count = line.rsplit('\t',3)[:2]
			if _name.count('\t') == 0:
				byBCNAME[_name] = int(_count)
			outfile.write('{0}\t{1}\n'.format(_name,'\t'*(2-_name.count('\t'))+_count))
			if re.match(order1re, line):
				name, cnt = line.decode('utf-8').split(u'\t')[:2]; cnt = int(cnt)
				if name in category1 or name in category2:
					covered += cnt
				else:
					uncovered += cnt
		fileinput.close()
	if show_coverage:
		print 'Coverage:\t', 1.*covered/(covered+uncovered)
	
	plt.figure(figsize=(10,10))
	subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.05)
	BCNAMEs, counts = zip(*sorted(byBCNAME.iteritems(),key=lambda x:x[1]))
	plt.barh(range(len(BCNAMEs)), counts, alpha=0.4)
	plt.yticks(range(len(BCNAMEs)), [_name.decode('utf-8') for _name in BCNAMEs])
	plt.title(u'各类事件投诉次数统计')
	plt.xlabel(u'投诉次数')
	ax = plt.gca()
	ax.xaxis.get_major_formatter().set_powerlimits((0,1))
	plt.savefig('./plots/event_category.png')


def statistic_by_time(filename, mode='week'):
	timeline_dict = {}
	for line in fileinput.input(filename):
		if re.match(order1re, line):
			name = line.split('\t')[0].decode('utf-8')
			timeline = map(lambda x:int(x),line.split('\t')[-2].split(','))
			timeline_dict[name] = timeline
	fileinput.close()

	if mode == 'week': 
		NT = 88; xlabel = u'周'
		for _plotname, _colnum, _category in [('12315市民服务热线管理',3,category1),('事部件词语',4,category2)]:
			plt.figure(figsize=(12,12))
			subplots_adjust(left=0.06,right=0.98,hspace=0.4,wspace=0.4)
			for i, name in enumerate(_category):
				subplot(_colnum,_colnum,i+1)
				plt.title(name)
				plt.plot(range(NT), timeline_dict[name][:NT], '*-', linewidth=1)
				plt.xlabel(xlabel); plt.ylabel(u'投诉次数')
			plt.savefig('./plots/time_{0}_{1}.png'.format(mode,_plotname))

	if mode == 'inwk': 
		NT = 7; xlabel = u'星期'
		for _plotname, _colnum, _category in [('12315市民服务热线管理',3,category1),('事部件词语',4,category2)]:
			plt.figure(figsize=(12,12))
			subplots_adjust(left=0.06,right=0.98,hspace=0.4,wspace=0.4)
			for i, name in enumerate(_category):
				subplot(_colnum,_colnum,i+1)
				plt.title(name)
				plt.bar(range(NT), timeline_dict[name][:NT], color='g', alpha=0.6)
				plt.xlabel(xlabel); plt.ylabel(u'投诉次数')
			plt.savefig('./plots/time_{0}_{1}.png'.format(mode,_plotname))


def statistic_by_space(filename):
	geo_dict = {}
	for line in fileinput.input(filename):
		if re.match(order1re, line):
			name = line.split('\t')[0].decode('utf-8')
			geo = np.array(map(lambda x:int(x),line.split('\t')[-1].split(',')))
			geo = geo.reshape(len(geo)**0.5,len(geo)**0.5)
			geo_dict[name] = geo
	fileinput.close()

	for _plotname, _colnum, _category in [('12315市民服务热线管理',3,category1),('事部件词语',4,category2)]:
		plt.figure(figsize=(20,20))
		subplots_adjust(left=0.06,right=0.98,hspace=0.4,wspace=0.4)
		for i, name in enumerate(_category):
			subplot(_colnum,_colnum,i+1)
			plt.title(name)
			(X, Y), C = meshgrid(np.arange(XNUM), np.arange(YNUM)), geo_dict[name][20:80,10:70]
			cset = pcolormesh(X, Y, C.T, cmap=cm.get_cmap("OrRd"))
			plt.axis([0, XNUM-1, 0, YNUM-1])
			plt.xticks(np.linspace(0,XNUM,5))
			plt.yticks(np.linspace(0,YNUM,5))
			plt.xlabel('X grid')
			plt.ylabel('Y grid')
			colorbar()
		plt.savefig('./plots/space_{0}.png'.format(_plotname))


if __name__ == '__main__':
	if sys.argv[1] == 'plot_geo_coordinates_compare':
		geo_coordinates_compare()
	if sys.argv[1] == 'statistic_inwk':
		NT = NT_INWK; statistic('../data/extracted.tsv',time_granularity='%w')
	if sys.argv[1] == 'statistic_week':
		NT = NT_WEEK; statistic('../data/extracted.tsv',time_granularity='%W')
	if sys.argv[1] == 'statistic_days':
		NT = NT_DAYS; statistic('../data/extracted.tsv',time_granularity='%j')
	if sys.argv[1] == 'plot_statistic_simplify':
		statistic_simplify('./statistic/CASE_HIERARCHY.txt', show_coverage=True)
	if sys.argv[1] == 'plot_statistic_by_time_inwk':
		statistic_by_time('./statistic/CASE_HIERARCHY.txt',mode='inwk')
	if sys.argv[1] == 'plot_statistic_by_time_week':
		statistic_by_time('./statistic/CASE_HIERARCHY.txt',mode='week')
	if sys.argv[1] == 'plot_statistic_by_space':
		XNUM, YNUM = 60, 60; statistic_by_space('./statistic/CASE_HIERARCHY.txt')

