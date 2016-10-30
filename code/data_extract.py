# -*- encoding:utf-8 -*-

import re
import glob
import time
import xlrd

from utils import to_unicode


def data_extr(dirname='../data/_original/网格中心业务数据/附件三/*', outfile='../data/extracted.tsv'):
	fieldextract = [u'TASKID',\
					u'COORDX',\
					u'COORDY',\
					u'INFOSOURCENAME',\
					u'DISCOVERTIME',\
					u'SOLVINGTIME',\
					u'ADDRESS',\
					u'STREETNAME',\
					u'DESCRIPTION',\
					u'EXECUTEDEPTNAME',\
					u'CASET_T.URGENTDEGREEWHEN1THEN\'紧急\'ELSE\'一般\'END',\
					u'CASET_T.USEREVALUATEWHEN0THEN\'满意\'WHEN1THEN\'基本满意\'WHEN2THEN\'不满意\'ELSE\'一般\'END',\
					u'INFOBCNAME',\
					u'INFOSCNAME',\
					u'INFOZCNAME',\
					u'CASEENDBCNAME',\
					u'CASEENDSCNAME']
	timeformater = lambda fieldname,field:field if not field else time.strftime('%Y/%m/%d %H:%M:%S',list(xlrd.xldate_as_tuple(field,0))+[0,0,0]) if fieldname.endswith(u'TIME') else field
	with open(outfile,'w') as outfile:
		for filename in glob.glob(dirname):
			print filename
			workbook = xlrd.open_workbook(filename)
			for index in range(workbook.nsheets):
				sheet = workbook.sheet_by_index(index)
				for lineno in range(sheet.nrows):
					if lineno % 10**4 == 0: sys.stdout.write(str(lineno)+'\r'); sys.stdout.flush()
					if lineno == 0:
						field_dict = {fieldname:i for i, fieldname in enumerate(sheet.row_values(lineno))}
					else:
						outfile.write(u'\t'.join([re.sub(ur'[\t\r\n]','',to_unicode(timeformater(fieldname,sheet.row_values(lineno)[field_dict[fieldname]]))) for fieldname in fieldextract]).encode('utf-8')+'\n')


if __name__ == '__main__':
	data_extr()

