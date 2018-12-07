# -*- coding:utf-8 -*-
import sys
import os
import simplejson
import nltk
# very slow
# from nltk.parse.corenlp import CoreNLPParser
from stanfordcorenlp import StanfordCoreNLP
import random
import re
import cPickle as pickle
import unicodedata
import numpy as np
from zhon.hanzi import punctuation
import gensim
from gensim import corpora, models
reload(sys)
sys.setdefaultencoding('utf8')

# stanford = StanfordCoreNLP(r'/home/admin/xinyx/Byte/stanford-corenlp-full-2018-01-31')
stanford = ''

def clean(data):
	data = unicodedata.normalize('NFKC', data.decode("UTF-8"))

	# '
	data = data.replace(u'‘', "'")  \
	.replace(u'’', "'") \
	.replace(u'′', "'") \
	.replace(u'‵', "'") \
	.replace(u'‛', "'") \
	.replace(u'`', "'")
	data = re.sub("&apos;", "'", data)

	# “ ” ‟ ″ ‶ `
	data = data.replace(u'“', '"') \
	.replace(u'”', '"') \
	.replace(u'‟', '"') \
	.replace(u'″', '"') \
	.replace(u'‶', '"')
	data = re.sub("&quot;", '"', data) 

	# delete all other chinese symbols
	data = re.sub(u"[%s]"%punctuation, ' ', data)

	# seperate two sentences
	data = re.sub("(\w[a-zA-Z]+)\.", "\\1. ", data)
	data = re.sub("(\s[a-zA-Z])\.(\S[^\.])", "\\1. \\2", data)
	data = re.sub("([0-9]+)\.([^0-9])", "\\1. \\2", data)
	data = re.sub("\?", "? ", data)
	## As for /, we seperate the word groud like km/h, and combine them in the predicted file
	data = re.sub("([^0-9])/", "\\1 / ", data)
	data = re.sub("/([^0-9])", " / \\1", data)

	# delete all tags
	# data = re.sub("&amp;", "&", data) 
	# data = re.sub("&apos;", "'", data) 
	# data = re.sub("&lt;", "<", data) 
	# data = re.sub("&gt;", ">", data) 
	# data = re.sub("&nbsp;", " ", data) 
	data = re.sub("&\w+;", " ", data)

	# replace number with #
	# data = re.sub("\#", ' ', data)
	# data = re.sub("\d+\.*\d*|\d*\.*\d+", '#', data)

	# delete all symbles except . , ? $ % : / and '
	data = re.sub(u"[\_\-\`\"\;\&\(\)\{\}\\\|\+©\=\*\!\@\[\]\~\^\<\> ̄·●•○]|''", ' ', data)

	# delete continous symbols
	data = re.sub('\.\.+', ' ', data)
	data = re.sub('\. \.[\. ]*?', ' ', data)
	data = re.sub('\:\:+', ' ', data)
	data = re.sub('\: \:[\: ]*?', ' ', data)
	data = re.sub('\,\,+', ',', data)
	data = re.sub(', ,[, ]*?', ',', data)
	data = re.sub('\?\?+', '?', data)
	data = re.sub('\? \?[\? ]*?', '?', data)
	data = re.sub('\$\$+', '?', data)
	data = re.sub('\$ \$[\$ ]*?', '?', data)
	data = re.sub('\/\/+', ' ', data)
	data = re.sub('\/ \/[\/ ]*?', ' ', data)
	data = re.sub('^[\.,;:\/\'\?% ]+', '', data)
	data = re.sub('[,;:\/ ]+$', '', data)

	# delete url
	## judge before entering this function

	# combine blanks
	data = ' '.join(data.split())

	return data

# 1. delete useless '
def clean_tokens(data):
	# data = re.sub("^' ", '', data)
	# data = re.sub(" '$", '', data)
	# data = re.sub(" ' ", ' ', data)
	return data

'''
0. 小写
1. 句末点号与下句要有空格
2. 去除&amp;等标记
3. 分割url
4. corenlp分词
5. 按照人的书写习惯简化修改标点
'''
def preprocess():
	global stanford
	stanford = StanfordCoreNLP(r'/home/admin/xinyx/Byte/stanford-corenlp-full-2018-01-31')
	data = []
	data += open("bytecup.corpus.train.0.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.1.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.2.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.3.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.4.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.5.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.6.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.7.txt").read().lower().strip().split('\n')
	data += open("bytecup.corpus.train.8.txt").read().lower().strip().split('\n')
	val = open("bytecup.corpus.validation_set.txt").read().lower().strip().split('\n')

	#data = clean(data).split('\n')
	#val = clean(val).split('\n')

	random.seed(100)
	random.shuffle(data)

	train = data[:-3000]
	myval = data[-3000:]

	train_c = open("train_content_clean.txt", 'w')
	train_t = open("train_title_clean.txt", 'w')
	myval_c = open("myval_content_clean.txt", 'w')
	myval_t = open("myval_title_clean.txt", 'w')
	val_c = open("val_content_clean.txt", 'w')
	val_t = open("val_title_clean.txt", 'w')

	num = 0
	for line in train:
		num += 1
		if num % 1000 == 0:
			print num
		try:
			txt = simplejson.loads(line)
			train_c.write(clean_tokens(' '.join(stanford.word_tokenize(clean(txt['content']))) + '\n'))
			train_t.write(clean_tokens(' '.join(stanford.word_tokenize(clean(txt['title']))) + '\n'))
		except:
			print line
			continue
	
	for line in myval:
		try:
			txt = simplejson.loads(line)
			myval_c.write(clean_tokens(' '.join(stanford.word_tokenize(clean(txt['content']))) + '\n'))
			myval_t.write(clean_tokens(' '.join(stanford.word_tokenize(clean(txt['title']))) + '\n'))
		except:
			print line
			continue

	for line in val:
		try:
			txt = simplejson.loads(line)
			val_c.write(clean_tokens(' '.join(stanford.word_tokenize(clean(txt['content']))) + '\n'))
			#val_t.write(clean_tokens(' '.join(stanford.word_tokenize(clean(txt['title']))) + '\n'))
		except:
			print line
			continue

def preprocess_test():
	global stanford
	stanford = StanfordCoreNLP(r'/home/admin/xinyx/Byte/stanford-corenlp-full-2018-01-31')
	data = open('bytecup.corpus.test_set.txt').read().strip().split('\n')
	fw = open('test_content.txt', 'w')
	res = []
	for idx,line in enumerate(data):
		try:
			txt = simplejson.loads(line)
			res.append((int(idx), clean_tokens(' '.join(stanford.word_tokenize(clean(txt['content']))))))
		except:
			print line
			continue
	res.sort()
	fw.write('\n'.join([str(idx+1) + ' ' + line for idx,line in res]))
	fw.close()


# truncated to the first n words
def postprocess(cs=-1, ts=-1, fs=[]):
	train_c = open("train_content4_nodup.txt", 'r').read().strip().split('\n')
	train_t = open("train_title4_nodup.txt", 'r').read().strip().split('\n')
	myval_c = open("myval_content4_nodup.txt", 'r').read().strip().split('\n')
	myval_t = open("myval_title4_nodup.txt", 'r').read().strip().split('\n')
	val_c = open("val_content4.txt", 'r').read().strip().split('\n')

	train_c_ = open("train_content4_nodup_%d.txt"%cs, 'w')
	train_t_ = open("train_title4_nodup_%d.txt"%ts, 'w')
	myval_c_ = open("myval_content4_nodup_%d.txt"%cs, 'w')
	myval_t_ = open("myval_title4_nodup_%d.txt"%ts, 'w')
	val_c_ = open("val_content4_%d.txt"%cs, 'w')
	
	if 'case' in fs:
		train_c_case = open('train_content_case4_nodup.txt', 'r').read().strip().split('\n')
		#train_t_case = open('train_title_case4.txt', 'r').read().strip().split('\n')
		train_c_case_ = open('train_content_case4_nodup_%d.txt'%cs, 'w')
		#train_t_case_ = open('train_title_case4_%d.txt'%ts, 'w')
		val_c_case = open("val_content_case4.txt", 'r').read().strip().split('\n')
		val_c_case_ = open("val_content_case4_%d.txt"%cs, 'w')
		myval_c_case = open("myval_content_case4_nodup.txt", 'r').read().strip().split('\n')
		myval_c_case_ = open("myval_content_case4_nodup_%d.txt"%cs, 'w')
		#myval_t_case = open("myval_title_case4.txt", 'r').read().strip().split('\n')
		#myval_t_case_ = open("myval_title_case4_%d.txt"%ts, 'w')
		for line in myval_c_case:
			myval_c_case_.write(' '.join(line.split(' ')[:cs]) + '\n')
		#for line in myval_t_case:
		#	myval_t_case_.write(' '.join(line.split(' ')[:ts]) + '\n')
		for line in train_c_case:
			train_c_case_.write(' '.join(line.split(' ')[:cs]) + '\n')
		#for line in train_t_case:
		#	train_t_case_.write(' '.join(line.split(' ')[:ts]) + '\n')
		for line in val_c_case:
			val_c_case_.write(' '.join(line.split(' ')[:cs]) + '\n')


	if 'pos' in fs:
		train_c_pos = open('train_content_pos4_nodup.txt', 'r').read().strip().split('\n')
		#train_t_pos = open('train_title_pos4.txt', 'r').read().strip().split('\n')
		train_c_pos_ = open('train_content_pos4_nodup_%d.txt'%cs, 'w')
		#train_t_pos_ = open('train_title_pos4_%d.txt'%ts, 'w')
		val_c_pos = open("val_content_pos4.txt", 'r').read().strip().split('\n')
		val_c_pos_ = open("val_content_pos4_%d.txt"%cs, 'w')
		myval_c_pos = open("myval_content_pos4_nodup.txt", 'r').read().strip().split('\n')
		myval_c_pos_ = open("myval_content_pos4_nodup_%d.txt"%cs, 'w')
		#myval_t_pos = open("myval_title_pos4.txt", 'r').read().strip().split('\n')
		#myval_t_pos_ = open("myval_title_pos4_%d.txt"%ts, 'w')
		for line in myval_c_pos:
			myval_c_pos_.write(' '.join(line.split(' ')[:cs]) + '\n')
		#for line in myval_t_pos:
		#	myval_t_pos_.write(' '.join(line.split(' ')[:ts]) + '\n')
		for line in val_c_pos:
			val_c_pos_.write(' '.join(line.split(' ')[:cs]) + '\n')
		for line in train_c_pos:
			train_c_pos_.write(' '.join(line.split(' ')[:cs]) + '\n')
		#for line in train_t_pos:
		#	train_t_pos_.write(' '.join(line.split(' ')[:ts]) + '\n')

	if 'ner' in fs:
		train_c_ner = open('train_content_ner4_nodup.txt', 'r').read().strip().split('\n')
		#train_t_ner = open('train_title_ner4.txt', 'r').read().strip().split('\n')
		train_c_ner_ = open('train_content_ner4_nodup_%d.txt'%cs, 'w')
		#train_t_ner_ = open('train_title_ner4_%d.txt'%ts, 'w')
		val_c_ner = open("val_content_ner4.txt", 'r').read().strip().split('\n')
		val_c_ner_ = open("val_content_ner4_%d.txt"%cs, 'w')
		myval_c_ner = open("myval_content_ner4_nodup.txt", 'r').read().strip().split('\n')
		myval_c_ner_ = open("myval_content_ner4_nodup_%d.txt"%cs, 'w')
		#myval_t_ner = open("myval_title_ner4.txt", 'r').read().strip().split('\n')
		#myval_t_ner_ = open("myval_title_ner4_%d.txt"%ts, 'w')
		for line in myval_c_ner:
			myval_c_ner_.write(' '.join(line.split(' ')[:cs]) + '\n')
		#for line in myval_t_ner:
		#	myval_t_ner_.write(' '.join(line.split(' ')[:ts]) + '\n')
		for line in val_c_ner:
			val_c_ner_.write(' '.join(line.split(' ')[:cs]) + '\n')
		for line in train_c_ner:
			train_c_ner_.write(' '.join(line.split(' ')[:cs]) + '\n')
		#for line in train_t_ner:
		#	train_t_ner_.write(' '.join(line.split(' ')[:ts]) + '\n')

	for line in train_c:
		train_c_.write(' '.join(line.split(' ')[:cs]) + '\n')
	for line in train_t:
		train_t_.write(' '.join(line.split(' ')[:ts]) + '\n')
	for line in myval_c:
		myval_c_.write(' '.join(line.split(' ')[:cs]) + '\n')
	for line in myval_t:
		myval_t_.write(' '.join(line.split(' ')[:ts]) + '\n')
	for line in val_c:
		val_c_.write(' '.join(line.split(' ')[:cs]) + '\n')

def postprocess_test(cs=-1, ts=-1, fs=[]):
	test_c = open("test_content4.txt", 'r').read().strip().split('\n')
	test_c_ = open("test_content4_%d.txt"%cs, 'w')
	
	if 'case' in fs:
		test_c_case = open("test_content_case4.txt", 'r').read().strip().split('\n')
		test_c_case_ = open("test_content_case4_%d.txt"%cs, 'w')
		for line in test_c_case:
			test_c_case_.write(' '.join(line.split(' ')[:cs]) + '\n')

	if 'pos' in fs:
		test_c_pos = open("test_content_pos4.txt", 'r').read().strip().split('\n')
		test_c_pos_ = open("test_content_pos4_%d.txt"%cs, 'w')
		for line in test_c_pos:
			test_c_pos_.write(' '.join(line.split(' ')[:cs]) + '\n')

	if 'ner' in fs:
		test_c_ner = open("test_content_ner4.txt", 'r').read().strip().split('\n')
		test_c_ner_ = open("test_content_ner4_%d.txt"%cs, 'w')
		for line in test_c_ner:
			test_c_ner_.write(' '.join(line.split(' ')[:cs]) + '\n')

	for line in test_c:
		test_c_.write(' '.join(line.split(' ')[:cs]) + '\n')

# 1. case features
# 2. ner
# 3. pos
def dumps(stanford, T, line, f, f_case, f_ner, f_pos):
	# only keeps first 500 words
	tokens = stanford.word_tokenize(line)[:500]
	# remove all symbols in content except & % and some ' .
	# delete all symbles except . , ? $ % : / and '
	line = ''
	for t in tokens:
		# content only contains $ and %
		if T == 'content' and t not in ['.', ',', '?', ':', '/', '\'']:
			line += t + ' '
		elif T == 'title' and t not in ['/', '\'']:
			line += t + ' '
	poses = stanford.pos_tag(line)
	# poses = [(token, pos) for (token, pos) in poses if token not in != ["'", '"']]
	# stanford ner is very slow
	# ners = stanford.ner(line)
	pos_ner = nltk.ne_chunk(nltk.pos_tag([token for (token,pos) in poses]))
	ts = []
	cases = []
	ps = []
	
	ners = []
	ns = []
	for ne in pos_ner:
		if type(ne) == nltk.tree.Tree:
			for item in ne:
				ners.append(ne.label())
		else:
			ners.append('O')
	
	i = -1
	for token,pos in poses:
		i += 1
		ts.append(token)
		ps.append(pos)
		ns.append(ners[i])
		if token.islower():
			cases.append('L')
		elif token.isupper():
			cases.append('U')
		elif token[0] >= 'A' and token[0] <= 'Z':
			cases.append('C')
		else:
			cases.append('N')
		n = len(token.split(' '))-1
		cases += [cases[-1]]*n
		ps += [ps[-1]]*n
		ns += [ns[-1]]*n

	ts = ' '.join(ts).lower()
	cases = ' '.join(cases)
	ps = ' '.join(ps)
	ns = ' '.join(ns)
	if len(ps.split(' ')) != len(ts.split(' ')) or len(ps.split(' ')) != len(cases.split(' ')) or len(ps.split(' ')) != len(ns.split(' ')):
		print "unaligned!\n" + line
		return -1

	f.write(ts + '\n')
	f_case.write(cases + '\n')
	f_pos.write(ps + '\n')
	f_ner.write(ns + '\n')

def preprocess_with_features():
	global stanford
	stanford = StanfordCoreNLP(r'/home/admin/xinyx/Byte/stanford-corenlp-full-2018-01-31')

	val = open("bytecup.corpus.validation_set.txt").read().strip().split('\n')


	train_c = open("train_content4.txt", 'w')
	train_c_case = open("train_content_case4.txt", 'w')
	train_c_ner = open("train_content_ner4.txt", 'w')
	train_c_pos = open("train_content_pos4.txt", 'w')

	train_t = open("train_title4.txt", 'w')
	train_t_case = open("train_title_case4.txt", 'w')
	train_t_ner = open("train_title_ner4.txt", 'w')
	train_t_pos = open("train_title_pos4.txt", 'w')
	
	myval_c = open("myval_content4.txt", 'w')
	myval_c_case = open("myval_content_case4.txt", 'w')
	myval_c_ner = open("myval_content_ner4.txt", 'w')
	myval_c_pos = open("myval_content_pos4.txt", 'w')
	
	myval_t = open("myval_title4.txt", 'w')
	myval_t_case = open("myval_title_case4.txt", 'w')
	myval_t_ner = open("myval_title_ner4.txt", 'w')
	myval_t_pos = open("myval_title_pos4.txt", 'w')
	
	val_c = open("val_content4.txt", 'w')
	val_c_case = open("val_content_case4.txt", 'w')
	val_c_ner = open("val_content_ner4.txt", 'w')
	val_c_pos = open("val_content_pos4.txt", 'w')

	num = 0
	for line in val:
		try:
			txt = simplejson.loads(line)
			dumps(stanford, 'content', clean(txt['content']), val_c, val_c_case, val_c_ner, val_c_pos)
			num += 1
			print num
		except:
			print line
			continue
	val_c.close()
	val_c_case.close()
	val_c_ner.close()
	val_c_pos.close()

	data = []
	data += open("bytecup.corpus.train.0.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.1.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.2.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.3.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.4.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.5.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.6.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.7.txt").read().strip().split('\n')
	data += open("bytecup.corpus.train.8.txt").read().strip().split('\n')

	data_ = []
	for line in data:
		txt = simplejson.loads(line)
		data_.append(txt['content'] + '@#$%%$#@' + txt['title'])

	print ('training size', len(data_))

	data_ = list(set(data_))
	random.seed(100)
	random.shuffle(data_)
	print ('after deduplicating...', len(data_))

	train = data_[:-3000]
	myval = data_[-3000:]
	num = 0
	for line in train:
		c,t = line.split('@#$%%$#@')
		if not pass_content_filter(c) or not pass_title_filter2(t):
			continue
		num += 1
		if num % 1000 == 0:
			print num
		try:
			if -1 != dumps(stanford, 'content', clean(c), train_c, train_c_case, train_c_ner, train_c_pos):
				if -1 == dumps(stanford, 'title', clean(t), train_t, train_t_case, train_t_ner, train_t_pos):
					dumps("fill", train_t, train_t_case, train_t_ner, train_t_pos)

		except:
			print line
			continue
	
	for line in myval:
		c,t = line.split('@#$%%$#@')
		if not pass_content_filter(c) or not pass_title_filter2(t):
			continue
		try:
			if -1 != dumps(stanford, 'content', clean(c), myval_c, myval_c_case, myval_c_ner, myval_c_pos):
				if -1 == dumps(stanford, 'title', clean(t), myval_t, myval_t_case, myval_t_ner, myval_t_pos):
					dumps("fill", myval_t, myval_t_case, myval_t_ner, myval_t_pos)

		except:
			print line
			continue


def preprocess_test_with_features():
	global stanford
	stanford = StanfordCoreNLP(r'/home/admin/xinyx/Byte/stanford-corenlp-full-2018-01-31')

	test = open("bytecup.corpus.test_set.txt").read().strip().split('\n')
	
	test_c = open("test_content4.txt", 'w')
	test_c_case = open("test_content_case4.txt", 'w')
	test_c_ner = open("test_content_ner4.txt", 'w')
	test_c_pos = open("test_content_pos4.txt", 'w')

	num = 0
	for line in test:
		try:
			txt = simplejson.loads(line)
			dumps(stanford, 'content', clean(txt['content']), test_c, test_c_case, test_c_ner, test_c_pos)
			num += 1
			print num
		except:
			print line
			continue
	test_c.close()
	test_c_case.close()
	test_c_ner.close()
	test_c_pos.close()

def raw_overlap(f, size=50):
	data = open(f).read().lower().strip().split('\n')
	data = data[:min(10000, len(data))]
	print len(data)

	Clens, Tlens = [], []
	overlaps = 0
	for index in range(len(data)):
		txt = simplejson.loads(data[index])
		C, T = nltk.word_tokenize(txt['content'])[:size], nltk.word_tokenize(txt['title'])
		Clens.append(len(C))
		Tlens.append(len(T))
		for word in T:
			if word in C:
				overlaps += 1

		if index % 1000 == 0:
			print index
			print 1.0*overlaps/sum(Tlens)
			print sum(Clens)/len(Clens), sum(Tlens)/len(Tlens)
	Clens.sort()
	Tlens.sort()
	print "content:", Clens[len(Clens)/4], Clens[len(Clens)/2], Clens[len(Clens)*3/4], sum(Clens)/len(Clens)
	print "title:", Tlens[len(Tlens)/4], Tlens[len(Tlens)/2], Tlens[len(Tlens)*3/4], sum(Tlens)/len(Tlens)
	print Clens[-1], Tlens[-1]

def adjust_pred_order():
	data = open("bytecup.corpus.test_set.txt").read().strip().split('\n')
	d = ['' for i in range(len((data)))]
	val = open("test_content4_fs_case_pos_200.txt").read().strip().split('\n')

	n = 0
	for line in data:
		id = int(simplejson.loads(line)['id'])
		d[id-1] = val[n]
		n += 1
	with open("test_content4_fs_case_pos_200_adj.txt", 'w') as fi:
		fi.write('\n'.join(d))


def combine_words_with_features(fs = []):
	train_c = re.sub(u'￨', '|', open("train_content4_nodup_300.txt").read().strip()).split('\n')
	train_t = re.sub(u'￨', '|', open("train_title4_nodup_20.txt").read().strip()).split('\n')
	myval_c = re.sub(u'￨', '|', open("myval_content4_nodup_300.txt").read().strip()).split('\n')
	myval_t = re.sub(u'￨', '|', open("myval_title4_nodup_20.txt").read().strip()).split('\n')
	val_c = re.sub(u'￨', '|', open("val_content4_300.txt").read().strip()).split('\n')

	if 'case' in fs:	
		train_c_case = open("train_content_case4_nodup_300.txt").read().strip().split('\n')
		#train_t_case = open("train_title_case4_nodup_20.txt").read().strip().split('\n')
		myval_c_case = open("myval_content_case4_nodup_300.txt").read().strip().split('\n')
		#myval_t_case = open("myval_title_case4_nodup_20.txt").read().strip().split('\n')
		val_c_case = open("val_content_case4_300.txt").read().strip().split('\n')
	
	if 'pos' in fs:
		train_c_pos = open("train_content_pos4_nodup_300.txt").read().strip().split('\n')
		#train_t_pos = open("train_title_pos4_nodup_20.txt").read().strip().split('\n')
		myval_c_pos = open("myval_content_pos4_nodup_300.txt").read().strip().split('\n')
		#myval_t_pos = open("myval_title_pos4_nodup_20.txt").read().strip().split('\n')
		val_c_pos = open("val_content_pos4_300.txt").read().strip().split('\n')

	if 'ner' in fs:
		train_c_ner = open("train_content_ner4_nodup_300.txt").read().strip().split('\n')
		#train_t_ner = open("train_title_ner4_nodup_20.txt").read().strip().split('\n')
		myval_c_ner = open("myval_content_ner4_nodup_300.txt").read().strip().split('\n')
		#myval_t_ner = open("myval_title_ner4_nodup_20.txt").read().strip().split('\n')
		val_c_ner = open("val_content_ner4_300.txt").read().strip().split('\n')

	if 'tfidf' in fs:
		train_c_ti = open("train_content_tfidf4_300.txt").read().strip().split('\n')
		myval_c_ti = open("myval_content_tfidf4_300.txt").read().strip().split('\n')
		val_c_ti = open("val_content_tfidf4_300.txt").read().strip().split('\n')
	
	
	train_c_fs = open('train_content4_nodup_fs_%s_300.txt'%'_'.join(fs), 'w')
	train_t_fs = open('train_title4_nodup_fs_%s_20.txt'%'_'.join(fs), 'w')
	for i in range(len(train_c)):
		# filter the nonsense data
		if len(train_c[i].split(' ')) < 100 or len(train_t[i].split(' ')) < 4:
			continue
		args_c, args_t = [train_c[i].split(' ')], [train_t[i].split(' ')]
		if 'case' in fs:
			args_c += [train_c_case[i].split(' ')]
			#args_t += [train_t_case[i].split(' ')]
		if 'pos' in fs:
			args_c += [train_c_pos[i].split(' ')]
			#args_t += [train_t_pos[i].split(' ')]
		if 'ner' in fs:
			args_c += [train_c_ner[i].split(' ')]
			#args_t += [train_t_ner[i].split(' ')]
		if 'tfidf' in fs:
			args_c += [train_c_ti[i].split(' ')]
			#args_t += [train_t_ner[i].split(' ')]
		
		train_c_fs.write(' '.join(['￨'.join(item) for item in zip(*args_c)]) + '\n')
		#train_t_fs.write(' '.join(['￨'.join(item) for item in zip(*args_t)]) + '\n')
		#title without fs
		train_t_fs.write(' '.join([item[0] for item in zip(*args_t)]) + '\n')

	myval_c_fs = open('myval_content4_nodup_fs_%s_300.txt'%'_'.join(fs), 'w')
	myval_t_fs = open('myval_title4_nodup_fs_%s_20.txt'%'_'.join(fs), 'w')
	for i in range(len(myval_c)):
		args_c, args_t = [myval_c[i].split(' ')], [myval_t[i].split(' ')]
		if 'case' in fs:
			args_c += [myval_c_case[i].split(' ')]
			#args_t += [myval_t_case[i].split(' ')]
		if 'pos' in fs:
			args_c += [myval_c_pos[i].split(' ')]
			#args_t += [myval_t_pos[i].split(' ')]
		if 'ner' in fs:
			args_c += [myval_c_ner[i].split(' ')]
			#args_t += [myval_t_ner[i].split(' ')]
		if 'tfidf' in fs:
			args_c += [myval_c_ti[i].split(' ')]
			#args_t += [myval_t_ner[i].split(' ')]
		
		myval_c_fs.write(' '.join(['￨'.join(item) for item in zip(*args_c)]) + '\n')
		#myval_t_fs.write(' '.join(['￨'.join(item) for item in zip(*args_t)]) + '\n')
		#title without fs
		myval_t_fs.write(' '.join([item[0] for item in zip(*args_t)]) + '\n')

	val_c_fs = open('val_content4_fs_%s_300.txt'%'_'.join(fs), 'w')
	for i in range(len(val_c)):
		args_c = [val_c[i].split(' ')]
		if 'case' in fs:
			args_c += [val_c_case[i].split(' ')]
		if 'pos' in fs:
			args_c += [val_c_pos[i].split(' ')]
		if 'ner' in fs:
			args_c += [val_c_ner[i].split(' ')]
		if 'tfidf' in fs:
			args_c += [val_c_ti[i].split(' ')]
		
		val_c_fs.write(' '.join(['￨'.join(item) for item in zip(*args_c)]) + '\n')

def combine_test_words_with_features(fs = []):
	test_c = re.sub(u'￨', '|', open("test_content4_300.txt").read().strip()).split('\n')

	if 'case' in fs:	
		test_c_case = open("test_content_case4_300.txt").read().strip().split('\n')
	
	if 'pos' in fs:
		test_c_pos = open("test_content_pos4_300.txt").read().strip().split('\n')

	if 'ner' in fs:
		test_c_ner = open("test_content_ner4_300.txt").read().strip().split('\n')

	if 'tfidf' in fs:
		test_c_ti = open("test_content_tfidf4_300.txt").read().strip().split('\n')
	
	
	test_c_fs = open('test_content4_fs_%s_300.txt'%'_'.join(fs), 'w')
	for i in range(len(test_c)):
		args_c = [test_c[i].split(' ')]
		if 'case' in fs:
			args_c += [test_c_case[i].split(' ')]
		if 'pos' in fs:
			args_c += [test_c_pos[i].split(' ')]
		if 'ner' in fs:
			args_c += [test_c_ner[i].split(' ')]
		if 'tfidf' in fs:
			args_c += [test_c_ti[i].split(' ')]
		
		test_c_fs.write(' '.join(['￨'.join(item) for item in zip(*args_c)]) + '\n')

def check_fs():
	data = open('train_content3_fs.txt').read().split('\n')
	for line in data:
		words_fs = line.split(' ')
		for w_f in words_fs:
			if len(w_f.split(u'￨')) != 2:
				print w_f
				print line
				exit(0)

def split_word_and_fs(fi, fo):
	data = open(fi).read().strip().split('\n')
	with open(fo, 'w') as FO:
		for line in data:
			FO.write(' '.join([item.split(u'￨')[0] for item in line.split(' ')]) + '\n')

def pass_content_filter(txt):
	if len(txt.split()) < 150:
		return False
	return True

def pass_title_filter(txt):
	if txt.find('http') >= 0:
		return False
	if txt.find('https') >= 0:
		return False
	txt = re.sub('\. [\. ]+', ' ', txt)
	txt = re.sub('\.\.+', ' ', txt)
	txt = re.sub('\: [\: ]+', ' ', txt)
	txt = re.sub(', [, ]+', ' , ', txt)
	txt = re.sub('\/ [\/ ]+', ' ', txt)
	txt = re.sub('^[\.,;:\/\']+', '', txt)
	txt = re.sub(' [\.,;:\/\']+$', '', txt)
	txt = ' '.join(txt.split())
	if len(txt.split()) < 4:
		return False
	return txt

def pass_title_filter2(txt):
	if txt.find('http') >= 0:
		return False
	if txt.find('https') >= 0:
		return False
	if len(txt.split()) < 4 or len(txt.split()) > 20:
		return False
	return True
	
def deduplicate():
	content_ = open("train_content4.txt").read().strip().split('\n')
	case_ = open("train_content_case4.txt").read().strip().split('\n')
	pos_ = open("train_content_pos4.txt").read().strip().split('\n')
	ner_ = open("train_content_ner4.txt").read().strip().split('\n')
	title_ = open("train_title4.txt").read().strip().split('\n')
	
	content_ += open("myval_content4.txt").read().strip().split('\n')
	case_ += open("myval_content_case4.txt").read().strip().split('\n')
	pos_ += open("myval_content_pos4.txt").read().strip().split('\n')
	ner_ += open("myval_content_ner4.txt").read().strip().split('\n')
	title_ += open("myval_title4.txt").read().strip().split('\n')
	
	# print (len(content_), len(case_), len(pos_), len(ner_), len(title_))
	assert(len(content_) == len(title_) and len(case_) == len(pos_) and len(pos_) == len(ner_) and len(content_) == len(ner_))

	print ('before deduplicating...', len(content_))
	
	n = 0
	mtitle = {}
	data = []
	content = []
	for i in range(len(title_)):
		if title_[i] in mtitle:
			mtitle[title_[i]].append(i)
			continue
		else:
			mtitle[title_[i]] = [i]
			data.append(content_[i] + 
					'@#$%%$#@' + title_[i] + 
					'@#$%%$#@' + case_[i] + 
					'@#$%%$#@' + pos_[i] + 
					'@#$%%$#@' + ner_[i])
	
	print ('after deduplicating...', len(data))

	content = open('train_content4_nodup.txt', 'w')
	title = open('train_title4_nodup.txt', 'w')
	case = open('train_content_case4_nodup.txt', 'w')
	pos = open('train_content_pos4_nodup.txt', 'w')
	ner = open('train_content_ner4_nodup.txt', 'w')

	val_content = open('myval_content4_nodup.txt', 'w')
	val_title = open('myval_title4_nodup.txt', 'w')
	val_case = open('myval_content_case4_nodup.txt', 'w')
	val_pos = open('myval_content_pos4_nodup.txt', 'w')
	val_ner = open('myval_content_ner4_nodup.txt', 'w')

	random.seed(100)
	random.shuffle(data)
	for line in data[:-2000]:
		cont, t, c, p, n = line.split('@#$%%$#@')
		content.write(cont + '\n')
		title.write(t + '\n')
		case.write(c + '\n')
		pos.write(p + '\n')
		ner.write(n + '\n')

	for line in data[-2000:]:
		cont, t, c, p, n = line.split('@#$%%$#@')
		val_content.write(cont + '\n')
		val_title.write(t + '\n')
		val_case.write(c + '\n')
		val_pos.write(p + '\n')
		val_ner.write(n + '\n')

	with open('dup.log', 'w') as fi:
		for k,v in mtitle.iteritems():
			if len(v) == 1:
				continue
			fi.write(k + '\n' + ' '.join(map(str, v)) + '\n')

	'''
	data = list(set(train + myval))
	random.seed(100)
	random.shuffle(data)
	print ('after deduplicating...', len(data))

	train_c = open("train_content2_fs_case_pos_ner_200_nodup.txt", 'w')
	train_t = open("train_title2_fs_case_pos_ner_20_nodup.txt", 'w')
	myval_c = open("myval_content2_fs_case_pos_ner_200_nodup.txt", 'w')
	myval_t = open("myval_title2_fs_case_pos_ner_20_nodup.txt", 'w')

	numt, numv = 0, 0
	for line in data[:-3000]:
		c,t = line.split('@#$%%$#@')
		if not pass_content_filter(c):
			continue
		t = pass_title_filter(t)
		if t != False:
			numt += 1
			#if numt == 95:
			#	print line
			#	print t
			#	exit(0)
			train_c.write(c + '\n')
			train_t.write(t + '\n')
	
	for line in data[-3000:]:
		c,t = line.split('@#$%%$#@')
		if not pass_content_filter(c): 
			continue
		t = pass_title_filter(t)
		if t != False:
			numv += 1
			myval_c.write(c + '\n')
			myval_t.write(t + '\n')

	print ('after filtering...', numt+numv)
	'''

def seperate_sentences():
	content_ = open('val_content4_.txt').read().strip().split('\n')
	case_ = open('val_content_case4_.txt').read().strip().split('\n')
	pos_ = open('val_content_pos4_.txt').read().strip().split('\n')
	ner_ = open('val_content_ner4_.txt').read().strip().split('\n')

	content = open('val_content4.txt', 'w')
	case = open('val_content_case4.txt', 'w')
	pos = open('val_content_pos4.txt', 'w')
	ner = open('val_content_ner4.txt', 'w')

	ncon, nc, np, nn = len(content_), len(case_), len(pos_), len(ner_)
	assert(ncon==nc and ncon==np and ncon==nn)

	for i in range(ncon):
		ts_, cs_, ps_, ns_ = content_[i].split(' '), case_[i].split(' '), pos_[i].split(' '), ner_[i].split(' ')
		ts, cs, ps, ns = [], [], [], []
		for j in range(len(ts_)):
			if None == re.search('\S\?\S', ts_[j]):
				ts.append(ts_[j])
				cs.append(cs_[j])
				ps.append(ps_[j])
				ns.append(ns_[j])
			else:
				ts_[j] = re.sub('(\S)\?(\S)', '\\1 ? \\2', ts_[j]).split(' ')
				ts += ts_[j]
				cs += [cs_[j]] * len(ts_[j])
				ps += [ps_[j]] * len(ts_[j])
				ns += [ns_[j]] * len(ts_[j])
		assert(len(ts)==len(cs) and len(cs)==len(ps) and len(ps)==len(ns))
		content.write(' '.join(ts) + '\n')
		case.write(' '.join(cs) + '\n')
		pos.write(' '.join(ps) + '\n')
		ner.write(' '.join(ns) + '\n')

def tfidf():
	documents = open("train_content4_nodup_200.txt", 'r').read().strip().split('\n') + \
				open("myval_content4_nodup_200.txt", 'r').read().strip().split('\n') + \
				open("val_content4_nodup_200.txt", 'r').read().strip().split('\n')
	
	res = []

	texts = [document.split(' ') for document in documents]
	word2idx =  corpora.Dictionary(texts)

	idx2word = {}
	for (index, word) in word2idx.items():
		idx2word[index] = word.decode('utf-8')

	corpus = [word2idx.doc2bow(text) for text in texts]
	model = models.TfidfModel(corpus)
	tfidf = model[corpus]


	for i in range(len(tfidf)):
		if i % 1000 == 0:
			print i
		ti_array = []
		word2tfidf = {}
		for (index, ti) in tfidf[i]:
			word2tfidf[idx2word[index]] = ti
		for word in texts[i]:
			ti_array.append(word2tfidf[word.decode('utf-8')])
		assert(len(ti_array) == len(texts[i]))
		res.append(ti_array)

	print('tfidf calcing done!')

	# discretize based on frequency
	K = 10

	res_ = []
	for r in res:
		res_ += r
	tfidf_sorted = sorted(res_)

	N = len(tfidf_sorted)/K
	splits = [tfidf_sorted[N*(i+1)] for i in range(K-1)]
	splits.append(1e12)
	print len(tfidf_sorted), splits

	res_k = []
	for li in res:
		ti_embs = []
		for ti in li:
			for i in range(K):
				if ti<splits[i]:
					ti_embs.append(i)
					break
		res_k.append(ti_embs)

	print ('decretizing done!')
	
	train = open('train_content_tfidf4_200.txt', 'w')
	val = open('myval_content_tfidf4_200.txt', 'w')
	test = open('val_content_tfidf4_200.txt', 'w')
	
	for li in res_k[-1000:]:
		test.write(' '.join(map(str, li)) + '\n')
	for li in res_k[-3000:-1000]:
		val.write(' '.join(map(str, li)) + '\n')
	for li in res_k[:-3000]:
		train.write(' '.join(map(str, li)) + '\n')

def toKaldi():
	train_c = open('train_content4_nodup_200.txt').read().strip().split('\t')
	train_t = open('train_title4_nodup_20.txt').read().strip().split('\t')
	myval_c = open('myval_content4_nodup_200.txt').read().strip().split('\t')
	myval_t = open('myval_title4_nodup_20.txt').read().strip().split('\t')
	val_c = open('val_content4_nodup_200_adj.txt').read().strip().split('\t')

	word2vec = {}
	glove = open('/home/admin/xinyx/data/nqg/sentence/data/glove.840B.300d.txt').read().strip().split('\n')
	print ('glove size', len(glove))
	for line in glove:
		word, vec = line.split(' ')[0], ' '.join(line.split(' ')[1:])
		word2vec[word] = vec
	
	ftc = open('train_content4_nodup_200.vec', 'w')
	ftc = open('train_content4_nodup_200.vec', 'w')
	ftc = open('train_content4_nodup_200.vec', 'w')
	ftc = open('train_content4_nodup_200.vec', 'w')
	ftc = open('train_content4_nodup_200.vec', 'w')
	for line in train_c:
		pass

def calc_feat_emb():
	case2emb = {}
	case2emb['U'] = np.asarray([1.,0.,0.,0.], 'float16')
	case2emb['C'] = np.asarray([0.,1.,0.,0.], 'float16')
	case2emb['L'] = np.asarray([0.,0.,1.,0.], 'float16')
	case2emb['N'] = np.asarray([0.,0.,0.,1.], 'float16')
	f = open('case2emb.pkl', 'wb')
	pickle.dump(case2emb, f, -1)
	f.close()

	tfidf2emb = {}
	tfidf2emb['0'] = np.asarray([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.], 'float16')
	tfidf2emb['1'] = np.asarray([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.], 'float16')
	tfidf2emb['2'] = np.asarray([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.], 'float16')
	tfidf2emb['3'] = np.asarray([0.,0.,0.,1.,0.,0.,0.,0.,0.,0.], 'float16')
	tfidf2emb['4'] = np.asarray([0.,0.,0.,0.,1.,0.,0.,0.,0.,0.], 'float16')
	tfidf2emb['5'] = np.asarray([0.,0.,0.,0.,0.,1.,0.,0.,0.,0.], 'float16')
	tfidf2emb['6'] = np.asarray([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.], 'float16')
	tfidf2emb['7'] = np.asarray([0.,0.,0.,0.,0.,0.,0.,1.,0.,0.], 'float16')
	tfidf2emb['8'] = np.asarray([0.,0.,0.,0.,0.,0.,0.,0.,1.,0.], 'float16')
	tfidf2emb['9'] = np.asarray([0.,0.,0.,0.,0.,0.,0.,0.,0.,1.], 'float16')
	f = open('tfidf2emb.pkl', 'wb')
	pickle.dump(tfidf2emb, f, -1)
	f.close()

	
	ners = open("train_content_ner4_nodup.txt", 'r').read().strip().split('\n')
	ners += open("myval_content_ner4_nodup.txt", 'r').read().strip().split('\n')
	ners += open("val_content_ner4_.txt", 'r').read().strip().split('\n')
	ners = [line.split() for line in ners]
	model = gensim.models.Word2Vec(ners, hs=1, negative=5, size=10, window=10, min_count=10, workers=8)
	ner_vocab = model.wv.vocab	
	ner2emb = {}
	for ner in ner_vocab:
		ner2emb[ner] = np.asarray(model[ner], 'float16')
	f = open('ner2emb.pkl', 'wb')
	pickle.dump(ner2emb, f, -1)
	f.close()

	poses  = open("train_content_pos4_nodup.txt", 'r').read().strip().split('\n')
	poses  += open("myval_content_pos4_nodup.txt", 'r').read().strip().split('\n')
	poses  += open("val_content_pos4_.txt", 'r').read().strip().split('\n')
	poses = [line.split() for line in poses]
	model = gensim.models.Word2Vec(poses, hs=1, negative=5, size=20, window=10, min_count=10, workers=8)
	pos_vocab = model.wv.vocab	
	pos2emb = {}
	for pos in pos_vocab:
		pos2emb[pos] = np.asarray(model[pos], 'float16')
	f = open('pos2emb.pkl', 'wb')
	pickle.dump(pos2emb, f, -1)
	f.close()


# raw_overlap("bytecup.corpus.train.0.txt", -1)
# preprocess_test()
# deduplicate()
# seperate_sentences()
# preprocess_with_features()
# preprocess_test_with_features()
# postprocess(200, 20, ['case','pos'])
# postprocess_test(300, 20, ['case','pos'])
# adjust_pred_order()
combine_words_with_features(['case', 'pos'])
# combine_test_words_with_features(['case', 'pos'])
# check_fs()
# split_word_and_fs('OpenNMT/pred.txt', 'pred.txt')
# tfidf()
# toKaldi()
# calc_feat_emb()

