# -*- encoding:utf-8 -*-

import re
import sys
import json
import math
import time
import fileinput
import numpy as np
from collections import defaultdict

from utils import *
from constants import *

if len(sys.argv) != 2:
	raise Exception('parameter wrong.')

if sys.argv[1].startswith('plot'):
	import matplotlib.pyplot as plt
	from pylab import *
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	mpl.rcParams['axes.unicode_minus'] = False


def statistic_description_length(filename):
	case_dict = defaultdict(list)
	
	def get_statistics(data):
		return u'{0}\t{1:.2f}\t{2}\t{3}\t{4}'.format(len(data),np.mean(data),np.median(data),np.min(data),np.max(data))

	for line in fileinput.input(filename):
		if fileinput.lineno() % 10**4 == 0: sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
		TASKID, COORDX, COORDY, INFOSOURCENAME, DISCOVERTIME, SOLVINGTIME, \
		ADDRESS, STREETNAME, DESCRIPTION, ENDRESULT, URGENTDEGREE, USEREVALUATE, \
		INFOBCNAME, INFOSCNAME, INFOZCNAME, CASEENDBCNAME, CASEENDSCNAME = map(lambda x:x.strip(), line.decode('utf-8').split(u'\t'))
		INFOBCNAME, INFOSCNAME, INFOZCNAME = map(lambda x:re.sub(ur'\(浦东\)','',x), [INFOBCNAME, INFOSCNAME, INFOZCNAME])
		if COORDX and COORDY and float(COORDX) and float(COORDY) and DISCOVERTIME:
			case_dict[INFOSOURCENAME].append(len(DESCRIPTION))
	fileinput.close()

	lens = []
	for INFOSOURCE, DESCRIPTIONlens in case_dict.iteritems():
		print u'{0}\t{1}'.format(INFOSOURCE, get_statistics(DESCRIPTIONlens))
		lens.extend(DESCRIPTIONlens)
	plt.figure(figsize=(12,5))
	plt.hist(lens, 100, range=(0,99), histtype='stepfilled', facecolor='g', alpha=0.6)
	plt.title(u'正文长度分布'); plt.xlabel(u'长度'); plt.ylabel(u'频次')
	plt.savefig('./statistic_clustering/description_length_statistics.png')


def statistic_description_byevent(filename, save=True):
	'''
		Generate meta data for clustering.
	'''

	import jieba
	# jieba.enable_parallel()
	from sklearn.externals import joblib
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.feature_extraction.text import CountVectorizer

	# TBD: generate stop words by annotation
	stopwords = set([u'，',u'。',u'（',u'）',u'：',u'的',u'了',u'上',u'在',u'米',u'有',u'乱'])
	filter_stopwords = lambda l:filter(lambda x:x not in stopwords,l)

	case_dict = defaultdict(list)
	# word_freq = defaultdict(int)

	for line in fileinput.input(filename):
		if fileinput.lineno() % 10**4 == 0: sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
		TASKID, COORDX, COORDY, INFOSOURCENAME, DISCOVERTIME, SOLVINGTIME, \
		ADDRESS, STREETNAME, DESCRIPTION, ENDRESULT, URGENTDEGREE, USEREVALUATE, \
		INFOBCNAME, INFOSCNAME, INFOZCNAME, CASEENDBCNAME, CASEENDSCNAME = map(lambda x:x.strip(), line.decode('utf-8').split(u'\t'))
		INFOBCNAME, INFOSCNAME, INFOZCNAME = map(lambda x:re.sub(ur'\(浦东\)','',x), [INFOBCNAME, INFOSCNAME, INFOZCNAME])
		# use ADDRESS and DESCRIPTION for semantic analysis
		if COORDX and COORDY and float(COORDX) and float(COORDY) and DISCOVERTIME:
			segments = filter_stopwords(jieba.cut(u'{0}\t{1}'.format(ADDRESS,DESCRIPTION)))
			# for word in segments: word_freq[word] += 1
			TEXT = re.sub(ur'\s+',' ',re.sub(ur'[^0-9a-zA-Z\u4e00-\u9fa5]',' ',' '.join(segments))).strip().lower()
			timeslot = int(time.strftime('%j',time.strptime(DISCOVERTIME,'%Y/%m/%d %H:%M:%S'))) + (366 if DISCOVERTIME.startswith('2016') else 0)
			case_dict[u'{0}\t{1}'.format(INFOBCNAME,INFOSCNAME)].append([TASKID,timeslot,float(COORDX),float(COORDY),TEXT])
	fileinput.close()

	# # word frequency statistics
	# for word, count in sorted(word_freq.iteritems(),key=lambda x:x[1],reverse=True)[:1000]:
	# 	print u'{0}\t{1}'.format(count, word).encode('utf-8')

	with open('models/statistic_description_byevent.json','w') as outfile: 
		outfile.write(json.dumps(case_dict))
	texts = []
	for INFONAME, TUPLE in case_dict.iteritems():
		texts.extend(map(lambda x:x[4], TUPLE))
	# TBD: use embeddings rather than tfidf to compute text similarities
	transformer, vectorizer = TfidfTransformer(), CountVectorizer(min_df=100)
	X = transformer.fit_transform(vectorizer.fit_transform(texts)).toarray()

	if save:
		joblib.dump(transformer, 'models/transformer.model')
		joblib.dump(vectorizer, 'models/vectorizer.model')
	else:
		return transformer, vectorizer


def event_mutual_information(filename):
	'''
		Compute mutual information between event categories.
	'''

	from sklearn.externals import joblib
	from sklearn.metrics import normalized_mutual_info_score

	case_dict = json.loads(open('models/statistic_description_byevent.json').read())
	try:
		transformer = joblib.load('models/transformer.model')
		vectorizer = joblib.load('models/vectorizer.model')
	except:
		transformer, vectorizer = statistic_description_byevent(filename, save=False)

	idxs, texts = [], []; vectors = []
	for idx, INFONAME in enumerate(sorted(case_dict)):
		idxs.extend([idx]*len(case_dict[INFONAME])); texts.extend(map(lambda x:x[4], case_dict[INFONAME]))
	idxs = np.array(idxs); X = transformer.transform(vectorizer.transform(texts)).toarray()
	for idx in range(len(case_dict.keys())):
		print '{0}/{1}'.format(idx,len(case_dict.keys()))
		if len(case_dict[case_dict.keys()[idx]])<10**2: continue
		vectors.append(X[idxs == idx][:10**4].sum(axis=0))
	
	with open('./statistic_clustering/event_category_mutual_information.tsv','w') as outfile:
		vectors = np.array(vectors); matrix = np.zeros((len(vectors),len(vectors)))
		for i in xrange(len(vectors)-1):
			print '{0}/{1}'.format(i,len(vectors))
			for j in xrange(i,len(vectors)):
				mutual_info = normalized_mutual_info_score(vectors[i],vectors[j])
				matrix[i,j] = matrix[j,i] = mutual_info
				if i != j and mutual_info >= 0.3:
					outfile.write(u'{0}\t{1}\t{2}\n'.format(mutual_info,re.sub(u'\t',u':',sorted(case_dict)[i]),re.sub(u'\t',u':',sorted(case_dict)[j])).encode('utf-8'))
	
	plt.figure(figsize=(14,12))
	(X, Y) = meshgrid(range(len(vectors)), range(len(vectors)))
	cset1 = pcolormesh(X, Y, matrix, cmap=cm.get_cmap("OrRd"))
	plt.xlim(0,len(vectors)-1); plt.ylim(0,len(vectors)-1)
	colorbar(cset1); plt.title('INFOZC Mutual Information'); plt.xlabel('INFOZCNAME'); plt.ylabel('INFOZCNAME')
	plt.savefig('./statistic_clustering/event_category_mutual_information.png')


def spatio_temporal_semantic_clustering(filename, mode='STREAMING', alg=None):
	'''
		Clustering by spatio, temporal and semantic meta data.
		TBD: consider cross category clustering (since different categories can have high mutual information)
	'''

	from sklearn.externals import joblib
	from sklearn.cluster import AgglomerativeClustering, DBSCAN

	complains = {}
	for line in fileinput.input('../data/extracted.tsv'):
		if fileinput.lineno() % 10**4 == 0: sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
		TASKID = line.decode('utf-8').split(u'\t')[0]
		complains[TASKID] = line.strip().decode('utf-8')
	fileinput.close()

	case_dict = json.loads(open('models/statistic_description_byevent.json').read())
	try:
		transformer = joblib.load('models/transformer.model')
		vectorizer = joblib.load('models/vectorizer.model')
	except:
		transformer, vectorizer = statistic_description_byevent(filename, save=False)

	def compute_similarity_time(t1,t2,thres=15,delta=7):
		if abs(t1-t2) >= thres:
			return 0
		else:
			return math.exp(-1.*abs(t1-t2)/delta)

	def compute_similarity_space(coor1,coor2,thres=1000,delta=500):
		distance = sum((coor1[i]-coor2[i])**2 for i in range(2))**0.5
		if distance >= thres:
			return 0
		else:
			return math.exp(-1.*distance/delta)

	def compute_similarity_texts(vec1,vec2,min_value=0.1):
		# TBD: should consider the text length impact
		cosine_similarity = (vec1*vec2).sum()/((vec1**2).sum()*(vec2**2).sum())**0.5
		return 0 if math.isnan(cosine_similarity) else max(cosine_similarity,min_value)

	with open('./statistic_clustering/results.tsv','w') as outfile:
		print '总共事件类型:\t', len(case_dict)
		
		if mode == 'BATCH':
			for idx, (INFONAME, LIST) in enumerate(sorted(case_dict.iteritems())):
				EVENTID = 1; INFOBCNAME,INFOSCNAME = INFONAME.split(u'\t')
				
				similarity_matrix = np.zeros((len(LIST),len(LIST)))
				taskids = np.array(map(lambda x:x[0], LIST))
				texts = map(lambda x:x[4], LIST); vecs = transformer.transform(vectorizer.transform(texts)).toarray()
				
				for i in xrange(len(LIST)):
					for j in xrange(i,len(LIST)):
						similarity_time = compute_similarity_time(LIST[i][1],LIST[j][1])
						similarity_space = 0 if not similarity_time else compute_similarity_space(LIST[i][2:4],LIST[j][2:4])
						similarity_texts = 0 if not similarity_time or not similarity_space else compute_similarity_texts(vecs[i],vecs[j])
						similarity_matrix[i,j] = similarity_matrix[j,i] = (similarity_time*similarity_space*similarity_texts)**(1./3)
				matrix_distance = 1./(similarity_matrix+0.1**5)

				for min_samples in range(2,10):
					if ALG == 'Agglomerative':
						clustering = AgglomerativeClustering(linkage='average', affinity='precomputed')
					elif ALG == 'DBSCAN':
						clustering = DBSCAN(eps=3, min_samples=min_samples, metric='precomputed', n_jobs=-1)
					else:
						raise Exception('alg not supported.')
					clustering.fit(matrix_distance)

					event_num = (clustering.labels_==-1).sum()+clustering.labels_.max()+1
					print u'\t{0}/{1}'.format(event_num,len(LIST)).encode('utf-8')
					if event_num >= 1.*len(LIST)/10: break
				
				for _cluster in range(clustering.labels_.max()+1):
					for taskid in taskids[clustering.labels_==_cluster]:
						outfile.write(u'{0}:{1}:{2}\t{3}\n'.format(INFOBCNAME,INFOSCNAME,EVENTID,complains[taskid]).encode('utf-8'))
					outfile.write('\n'); EVENTID += 1

			for taskid in taskids[clustering.labels_==-1]:
				outfile.write(u'{0}:{1}:{2}\t{3}\n'.format(INFOBCNAME,INFOSCNAME,EVENTID,complains[taskid]).encode('utf-8'))
				outfile.write('\n'); EVENTID += 1

		elif mode == 'STREAMING':
			for idx, (INFONAME, LIST) in enumerate(sorted(case_dict.iteritems())):
				EVENTID = 1; INFOBCNAME,INFOSCNAME = INFONAME.split(u'\t')
				# print u'{0}\t{1}:{2}\t{3}'.format(idx,INFOBCNAME,INFOSCNAME,len(LIST)).encode('utf-8')
				
				clusters = []; taskids = np.array(map(lambda x:x[0], LIST))
				texts = map(lambda x:x[4], LIST); vecs = transformer.transform(vectorizer.transform(texts)).toarray()

				for i in xrange(len(LIST)):
					if i%100 == 0 or i == len(LIST)-1:
						sys.stdout.write(str(int(100*(i+1)/len(LIST)))+'% \r'); sys.stdout.flush()
					similar_clusters = []; threshold = 0.2
					
					for c in xrange(len(clusters)-1,-1,-1):
						similarity_time = compute_similarity_time(LIST[i][1],clusters[c]['time'].mean(axis=0))
						if not similarity_time: break
						similarity_space = compute_similarity_space(LIST[i][2:4],clusters[c]['space'].mean(axis=0))
						if not similarity_space: continue
						similarity_texts = compute_similarity_texts(vecs[i],clusters[c]['texts'])
						if not similarity_texts: continue
						similarity = (similarity_time*similarity_space*similarity_texts)**(1./3)
						if similarity >= threshold: similar_clusters.append((similarity,c))
					
					if not similar_clusters:
						clusters.append({
							'id':[taskids[i]],
							'time':np.array([LIST[i][1]]),
							'space':np.array([LIST[i][2:4]]),
							'texts':vecs[i]
						})
					else:
						most_similar = sorted(similar_clusters,reverse=True)[0][1]
						clusters[most_similar]['id'].append(taskids[i])
						clusters[most_similar]['time'] = np.hstack([clusters[most_similar]['time'],LIST[i][1]])
						clusters[most_similar]['space'] = np.vstack([clusters[most_similar]['space'],LIST[i][2:4]])
						clusters[most_similar]['texts'] += vecs[i]
				
				print u'\n{0}\t{1}:{2}\t{3}/{4}'.format(idx,INFOBCNAME,INFOSCNAME,len(clusters),len(LIST)).encode('utf-8')

				for c in xrange(len(clusters)):
					# print len(clusters[c]['id']), clusters[c]['time'].mean(axis=0), clusters[c]['space'].mean(axis=0)
					for taskid in clusters[c]['id']:
						outfile.write(u'{0}:{1}:{2}\t{3}\n'.format(INFOBCNAME,INFOSCNAME,c+1,complains[taskid]).encode('utf-8'))
					outfile.write('\n')

		else:
			raise Exception('mode not supported.')


def clustering_result_analysis(filename):
	events = defaultdict(list); event = []

	get_taskids = lambda l:map(lambda x:x[0],l)
	get_cords = lambda l:None # TBD
	get_times = lambda l:None # TBD

	for line in fileinput.input(filename):
		if fileinput.lineno() % 10**4 == 0: sys.stdout.write(str(fileinput.lineno())+'\r'); sys.stdout.flush()
		if line.strip():
			EVENTNAME, TASKID, COORDX, COORDY, INFOSOURCENAME, DISCOVERTIME, SOLVINGTIME, \
			ADDRESS, STREETNAME, DESCRIPTION, ENDRESULT, URGENTDEGREE, USEREVALUATE = line.strip().decode('utf-8').split(u'\t')[:13]
			INFOBCNAME, INFOSCNAME, EVENTID = EVENTNAME.split(u':')
			COORDX, COORDY, DISCOVERTIME = float(COORDX), float(COORDY), int(time.strftime('%j',time.strptime(DISCOVERTIME,'%Y/%m/%d %H:%M:%S'))) + (366 if DISCOVERTIME.startswith('2016') else 0)
			event.append((TASKID,COORDX,COORDY,DISCOVERTIME))
		else:
			events[INFOBCNAME].append(get_taskids(event)); event = []
	fileinput.close()

	plt.figure(figsize=(10,10))
	subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.05)
	BCNAMEs, counts = zip(*sorted([(BCNAME,len(eventlist)) for BCNAME,eventlist in events.iteritems()],key=lambda x:x[1]))
	plt.barh(range(len(BCNAMEs)), counts, alpha=0.4)
	plt.yticks(range(len(BCNAMEs)), BCNAMEs)
	plt.title(u'各类事件次数统计')
	plt.xlabel(u'频次')
	ax = plt.gca()
	ax.xaxis.get_major_formatter().set_powerlimits((0,1))
	plt.savefig('./statistic_clustering/event_category.png')

	with open('./statistic_clustering/clustering_eventids.tsv','w') as outfile:
		for BCNAME, eventlist in events.iteritems():
			if not BCNAME: continue
			for eid, ids in enumerate(eventlist):
				outfile.write(u'{0}\t{1}\t{2}\n'.format(BCNAME,eid,u','.join(ids)).encode('utf-8'))


if __name__ == '__main__':
	if sys.argv[1] == 'plot_statistic_description_length':
		statistic_description_length('../data/extracted.tsv')
	if sys.argv[1] == 'statistic_description_byevent':
		statistic_description_byevent('../data/extracted.tsv')
	if sys.argv[1] == 'plot_event_mutual_information':
		event_mutual_information('../data/extracted.tsv')
	if sys.argv[1] == 'spatio_temporal_semantic_clustering':
		spatio_temporal_semantic_clustering('../data/extracted.tsv')
	if sys.argv[1] == 'plot_clustering_result_analysis':
		clustering_result_analysis('./statistic_clustering/results.tsv')

