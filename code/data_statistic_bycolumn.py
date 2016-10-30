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


def statistic(filename, bycol='INFOSOURCENAME', outdir='statistic_infosource', time_granularity='%m', Nmonth=20):
	# TBD: use collections.defaultdict()
	case_dict = {}
	
	for line in fileinput.input(filename):
		if fileinput.lineno() % 10**4 == 0: print sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
		TASKID, COORDX, COORDY, INFOSOURCENAME, DISCOVERTIME, SOLVINGTIME, \
		ADDRESS, STREETNAME, DESCRIPTION, EXECUTEDEPTNAME, URGENTDEGREE, USEREVALUATE, \
		INFOBCNAME, INFOSCNAME, INFOZCNAME, CASEENDBCNAME, CASEENDSCNAME = map(lambda x:x.strip(), line.decode('utf-8').split(u'\t'))
		INFOBCNAME, INFOSCNAME, INFOZCNAME = map(lambda x:re.sub(ur'\(浦东\)','',x), [INFOBCNAME, INFOSCNAME, INFOZCNAME])
		if COORDX and COORDY and float(COORDX) and float(COORDY) and DISCOVERTIME:
			timeslot = time.strptime(DISCOVERTIME,'%Y/%m/%d %H:%M:%S').tm_mon + (12 if DISCOVERTIME.startswith('2016') else 0)
			if bycol == 'STREETNAME':
				COL = STREETNAME
			elif bycol == 'EXECUTEDEPTNAME':
				COL = EXECUTEDEPTNAME
			elif bycol == 'INFOSOURCENAME':
				COL = INFOSOURCENAME
			else:
				raise Exception('Column is not supported.')
			case_dict[COL] = case_dict.get(COL,{})
			case_dict[COL][INFOBCNAME] = case_dict[COL].get(INFOBCNAME,[0]*24)
			case_dict[COL][INFOBCNAME][timeslot-1] += 1
	fileinput.close()

	compare = lambda array,delta:[u'{0:+.2f}'.format(1.*(array[i]-array[i-delta])/array[i-delta]) if i-delta>=0 and array[i-delta] else u'-' for i in xrange(len(array))]
	for i, BCNAME in enumerate(category1+category2):
		with open('{0}/{1}{2}.txt'.format(outdir,i,BCNAME.encode('utf-8')),'w') as outfile:
			outfile.write(u'{0}\t{1}\n'.format(bycol.decode('utf-8'),u'\t'.join([u'{0}年{1}月({2})'.format(year,month,title) for year in (2015,2016) for month in xrange(1,13) for title in (u'次数',u'对比上月',u'对比上年同期')][:Nmonth*3])).encode('utf-8'))
			for COL, cases in case_dict.iteritems():
				if BCNAME in cases and BCNAME:
					outfile.write(u'{0}\t{1}\n'.format(COL,u'\t'.join([u'{0}\t{1}\t{2}'.format(c1,c2,c3) for c1,c2,c3 in zip(cases[BCNAME], compare(cases[BCNAME],1), compare(cases[BCNAME],12))][:Nmonth])).encode('utf-8'))

	with open('{0}/_event.txt'.format(outdir,bycol),'w') as outfile:
		outfile.write(bycol+'\t'+'\t'.join(BCNAME.encode('utf-8') for _, BCNAME in enumerate(category1+category2))+'\n')
		for COL, cases in case_dict.iteritems():
			outfile.write(COL.encode('utf-8')+'\t'+'\t'.join([str(sum(case_dict[COL].get(BCNAME,[0]))) for BCNAME in category1+category2])+'\n')

	with open('{0}/_sumup.txt'.format(outdir,bycol),'w') as outfile:
		outfile.write(bycol+'\t'+'事件总数'+'\n')
		for COL, cases in case_dict.iteritems():
			outfile.write(COL.encode('utf-8')+'\t'+str(sum([sum(case_dict[COL].get(BCNAME,[0])) for BCNAME in category1+category2]))+'\n')


def select_important(filename, field='INFOSOURCENAME', values=[u'领导交办',u'媒体关注',u'市级督查'], outfilename='../data/selected.tsv'):
	with open(outfilename,'w') as outfile:
		for line in fileinput.input(filename):
			if fileinput.lineno() % 10**4 == 0: print sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
			TASKID, COORDX, COORDY, INFOSOURCENAME, DISCOVERTIME, SOLVINGTIME, \
			ADDRESS, STREETNAME, DESCRIPTION, EXECUTEDEPTNAME, URGENTDEGREE, USEREVALUATE, \
			INFOBCNAME, INFOSCNAME, INFOZCNAME, CASEENDBCNAME, CASEENDSCNAME = map(lambda x:x.strip(), line.decode('utf-8').split(u'\t'))
			if field == 'INFOSOURCENAME':
				if INFOSOURCENAME in values:
					outfile.write(line)
		fileinput.close()


if __name__ == '__main__':
	if sys.argv[1] == 'statistic_STREETNAME':
		statistic('../data/extracted.tsv',bycol='STREETNAME',outdir='statistic_street',time_granularity='%W')
	if sys.argv[1] == 'statistic_EXECUTEDEPTNAME':
		statistic('../data/extracted.tsv',bycol='EXECUTEDEPTNAME',outdir='statistic_executedept',time_granularity='%W')
	if sys.argv[1] == 'statistic_INFOSOURCENAME':
		statistic('../data/extracted.tsv',bycol='INFOSOURCENAME',outdir='statistic_infosource',time_granularity='%W')
	if sys.argv[1] == 'select_important':
		select_important('../data/extracted.tsv')

