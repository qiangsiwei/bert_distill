# -*- coding: utf-8 -*-

import jieba, random, fileinput, numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

def load_data(name):
	def get_w2v():
		for line in open('data/cache/word2vec').read().strip().split('\n'):
			line = line.strip().split()
			if not line: continue
			yield line[0],np.array(list(map(float,line[1:])))
	tokenizer = Tokenizer(filters='',lower=True,split=' ',oov_token=1)
	texts = [' '.join(jieba.cut(line.split('\t',1)[1].strip()))\
		for line in open('data/{}/{}.txt'.format(name,name)
			).read().strip().split('\n')]
	tokenizer.fit_on_texts(texts)
	# with open('word2vec','w') as out:
	# 	for line in fileinput.input('sgns.sogou.word'):
	# 		word = line.strip().split()[0]
	# 		if word in tokenizer.word_index:
	# 			out.write(line+'\n')
	# 	fileinput.close()
	x_train,y_train = [],[]; text_train = []
	for line in open('data/{}/train.txt'.format(name)).read().strip().split('\n'):
		label,text = line.split('\t',1)
		text_train.append(text.strip())
		x_train.append(' '.join(jieba.cut(text.strip())))
		y_train.append(int(label))
	x_train = tokenizer.texts_to_sequences(x_train)
	x_dev,y_dev = [],[]; text_dev = []
	for line in open('data/{}/dev.txt'.format(name)).read().strip().split('\n'):
		label,text = line.split('\t',1)
		text_dev.append(text.strip())
		x_dev.append(' '.join(jieba.cut(text.strip())))
		y_dev.append(int(label))
	x_dev = tokenizer.texts_to_sequences(x_dev)
	x_test,y_test = [],[]; text_test = []
	for line in open('data/{}/test.txt'.format(name)).read().strip().split('\n'):
		label,text = line.split('\t',1)
		text_test.append(text.strip())
		x_test.append(' '.join(jieba.cut(text.strip())))
		y_test.append(int(label))
	x_test = tokenizer.texts_to_sequences(x_test)
	v_size = len(tokenizer.word_index)+1
	embs,w2v = np.zeros((v_size,300)),dict(get_w2v())
	for word,index in tokenizer.word_index.items():
		if word in w2v: embs[index] = w2v[word]
	return (x_train,y_train,text_train),\
		   (x_dev,y_dev,text_dev),\
		   (x_test,y_test,text_test),\
		   v_size,embs   

if __name__ == '__main__':
	load_data(name='hotel')
