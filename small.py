# -*- coding: utf-8 -*-

import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from keras.preprocessing import sequence
from utils import load_data

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

class RNN(nn.Module):
	def __init__(self, x_dim, e_dim, h_dim, o_dim): 
		super(RNN,self).__init__()
		self.h_dim = h_dim
		self.dropout = nn.Dropout(0.2)
		self.emb = nn.Embedding(x_dim,e_dim,padding_idx=0)
		self.lstm = nn.LSTM(e_dim,h_dim,bidirectional=True,batch_first=True)
		self.fc = nn.Linear(h_dim*2,o_dim)
		self.softmax = nn.Softmax(dim=1)
		self.log_softmax = nn.LogSoftmax(dim=1)
	def forward(self, x, lens):
		embed = self.dropout(self.emb(x))
		out,_ = self.lstm(embed)
		hidden = self.fc(out[:,-1,:])
		return self.softmax(hidden), self.log_softmax(hidden)

class CNN(nn.Module):
	def __init__(self, x_dim, e_dim, h_dim, o_dim): 
		super(CNN,self).__init__()
		self.emb = nn.Embedding(x_dim,e_dim,padding_idx=0)
		self.dropout = nn.Dropout(0.2)
		self.conv1 = nn.Conv2d(1,h_dim,(3,e_dim))
		self.conv2 = nn.Conv2d(1,h_dim,(4,e_dim))
		self.conv3 = nn.Conv2d(1,h_dim,(5,e_dim))
		self.fc = nn.Linear(h_dim*3,o_dim)
		self.softmax = nn.Softmax(dim=1)
		self.log_softmax = nn.LogSoftmax(dim=1)
	def forward(self, x, lens):
		embed = self.dropout(self.emb(x)).unsqueeze(1)
		c1 = torch.relu(self.conv1(embed).squeeze(3))
		p1 = torch.max_pool1d(c1,c1.size()[2]).squeeze(2)
		c2 = torch.relu(self.conv2(embed).squeeze(3))
		p2 = torch.max_pool1d(c2,c2.size()[2]).squeeze(2)
		c3 = torch.relu(self.conv3(embed).squeeze(3))
		p3 = torch.max_pool1d(c3,c3.size()[2]).squeeze(2)
		pool = self.dropout(torch.cat((p1,p2,p3),1))
		hidden = self.fc(pool)
		return self.softmax(hidden), self.log_softmax(hidden)

class Model(object):
	def __init__(self, v_size):
		self.model = None
		self.b_size = 64
		self.lr = 0.001
		self.model = RNN(v_size,256,256,2)
		# self.model = CNN(v_size,256,128,2)
	def train(self, x_tr, y_tr, l_tr, x_te, y_te, l_te, epochs=15):
		assert self.model is not None
		if USE_CUDA: self.model = self.model.cuda()
		loss_func = nn.NLLLoss()
		opt = optim.Adam(self.model.parameters(),lr=self.lr)
		for epoch in range(epochs):
			losses = []; accu = []
			self.model.train()
			for i in range(0,len(x_tr),self.b_size):
				self.model.zero_grad()
				bx = Variable(LTensor(x_tr[i:i+self.b_size]))
				by = Variable(LTensor(y_tr[i:i+self.b_size]))
				bl = Variable(LTensor(l_tr[i:i+self.b_size]))
				_,py = self.model(bx,bl)
				loss = loss_func(py,by)
				loss.backward(); opt.step()
				losses.append(loss.item())
			self.model.eval()
			with torch.no_grad():
				for i in range(0,len(x_te),self.b_size):
					bx = Variable(LTensor(x_te[i:i+self.b_size]))
					by = Variable(LTensor(y_te[i:i+self.b_size]))
					bl = Variable(LTensor(l_te[i:i+self.b_size]))
					_,py = torch.max(self.model(Variable(LTensor(bx)),bl)[1],1)
					accu.append((py==by).float().mean().item())
			print(np.mean(losses),np.mean(accu))

if __name__ == '__main__':
	x_len = 50
	# ----- ----- ----- ----- -----
	# from keras.datasets import imdb
	# v_size = 10000
	# (x_tr,y_tr),(x_te,y_te) = imdb.load_data(num_words=v_size)
	# ----- ----- ----- ----- -----
	name = 'hotel' # clothing, fruit, hotel, pda, shampoo
	(x_tr,y_tr,_),_,(x_te,y_te,_),v_size,_ = load_data(name)
	l_tr = list(map(lambda x:min(len(x),x_len),x_tr))
	l_te = list(map(lambda x:min(len(x),x_len),x_te))
	x_tr = sequence.pad_sequences(x_tr,maxlen=x_len)
	x_te = sequence.pad_sequences(x_te,maxlen=x_len)
	clf = Model(v_size)
	clf.train(x_tr,y_tr,l_tr,x_te,y_te,l_te)
