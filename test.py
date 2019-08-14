# -*-coding:utf-8-*-

import pickle, numpy as np
from keras.layers import *
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from utils import load_data

def get_textcnn(x_len, v_size, embs):
	x = Input(shape=(x_len,),dtype='int32')
	# embed = Embedding(v_size,300)(x)
	embed = Embedding(v_size,300,embeddings_initializer=Constant(embs),trainable=False)(x)
	cnn1 = Convolution1D(256,3,padding='same',strides=1,activation='relu')(embed)
	cnn1 = MaxPool1D(pool_size=4)(cnn1)
	cnn2 = Convolution1D(256,4,padding='same',strides=1,activation='relu')(embed)
	cnn2 = MaxPool1D(pool_size=4)(cnn2)
	cnn3 = Convolution1D(256,5,padding='same',strides=1,activation='relu')(embed)
	cnn3 = MaxPool1D(pool_size=4)(cnn3)
	cnn = concatenate([cnn1,cnn2,cnn3],axis=-1)
	flat = Flatten()(cnn)
	drop = Dropout(0.2,name='drop')(flat)
	y = Dense(2,activation='softmax')(drop)
	model = Model(inputs=x,outputs=y)
	return model

def get_birnn(x_len, v_size, embs):
	x = Input(shape=(x_len,),dtype='int32')
	# embed = Embedding(v_size,300)(x)
	embed = Embedding(v_size,300,embeddings_initializer=Constant(embs),trainable=False)(x)
	# bi = Bidirectional(GRU(256,activation='tanh',recurrent_dropout=0.2,dropout=0.2,return_sequences=True))(embed)
	bi = Bidirectional(GRU(256,activation='tanh',recurrent_dropout=0.2,dropout=0.2))(embed)
	y = Dense(2,activation='softmax')(bi)
	model = Model(inputs=x,outputs=y)
	return model

def run_small():
	x_len = 50
	name = 'hotel' # clothing, fruit, hotel, pda, shampoo
	(x_tr,y_tr,_),_,(x_te,y_te,_),v_size,embs = load_data(name)
	x_tr = sequence.pad_sequences(x_tr,maxlen=x_len)
	x_te = sequence.pad_sequences(x_te,maxlen=x_len)
	y_tr = to_categorical(y_tr,2)
	y_te = to_categorical(y_te,2)
	# model = get_textcnn(x_len,v_size,embs)
	model = get_birnn(x_len,v_size,embs)
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.fit(x_tr,y_tr,batch_size=32,epochs=10,validation_data=(x_te,y_te))

def run_distill():
	x_len = 50

	# ----- ----- ----- ----- -----
	# from keras.datasets import imdb
	# (x_tr,y_tr),(x_te,y_te) = imdb.load_data(num_words=10000)
	# ----- ----- ----- ----- -----

	name = 'hotel' # clothing, fruit, hotel, pda, shampoo
	(x_tr,y_tr,_),(x_de,y_de,_),(x_te,y_te,_),v_size,embs = load_data(name)
	x_tr = sequence.pad_sequences(x_tr,maxlen=x_len)
	x_de = sequence.pad_sequences(x_de,maxlen=x_len)
	x_te = sequence.pad_sequences(x_te,maxlen=x_len)
	y_tr = to_categorical(y_tr,2)
	y_de = to_categorical(y_de,2)
	y_te = to_categorical(y_te,2)
	with open('data/cache/t_tr','rb') as fin: y_tr = pickle.load(fin)
	with open('data/cache/t_de','rb') as fin: y_de = pickle.load(fin)
	# y_tr = to_categorical(y_tr.argmax(axis=1),2)
	# y_de = to_categorical(y_de.argmax(axis=1),2)

	# ----- ----- distill ----- -----
	# model = get_textcnn(x_len,v_size,embs)
	model = get_birnn(x_len,v_size,embs)
	x_tr = np.vstack([x_tr,x_de])
	y_tr = np.vstack([y_tr,y_de])
	model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.fit(x_tr,y_tr,batch_size=32,epochs=10,validation_data=(x_te,y_te))
	# ----- ----- ----- ----- -----

if __name__ == '__main__':
	# run_small()
	run_distill()
